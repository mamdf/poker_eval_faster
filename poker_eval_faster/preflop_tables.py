from __future__ import annotations

from functools import cache
from math import comb
from pathlib import Path
from typing import Sequence

import numpy as np

from .main import (
    _THREE_WAY_EQUITY_WEIGHTS,
    _simulation_cards,
    _simulation_integer,
    canonical_combo_masks,
    combo_to_hand_class,
)
from .preflop_canonical import _preflop_canonical_counts, _preflop_range_matchup_profile
from .three_way_class_lookup import HAND_CLASSES_169, class_combo_ids, class_label_to_id, packed_class_triple_index


PREFLOP_VS_RANDOM_FILE = "preflop_equity_vs_random.npy"
PREFLOP_HU_FILE = "preflop_hu_class_equity.npy"
PREFLOP_THREE_WAY_FILE = "preflop_3way_class_equity.npy"
PREFLOP_MAX_OPPONENTS = 9
PREFLOP_EXACT_OPPONENTS = 2
PREFLOP_RANDOM_SAMPLES = 20_000_000

_DATA_DIR = Path(__file__).parent / "data"
_NUM_CLASSES = len(HAND_CLASSES_169)
# 13 order counts -> six times each seat's pot share, so derivations stay integral.
_THREE_WAY_WEIGHTS_X6 = np.rint(np.array(_THREE_WAY_EQUITY_WEIGHTS) * 6).astype(np.int64)


def preflop_random_seed(num_opponents: int, class_id: int) -> int:
    """Seed of the Monte Carlo estimate stored for one class and opponent count."""
    return 1_000_000 * int(num_opponents) + int(class_id)


@cache
def _table(name: str) -> np.ndarray:
    path = _DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; build it with scripts/build_preflop_equity_tables.py")
    return np.load(path, mmap_mode="r" if name == PREFLOP_THREE_WAY_FILE else None, allow_pickle=False)


def _class_id(label: str) -> int:
    if not isinstance(label, str):
        raise ValueError(f"Hand class must be a label like 'AKs', got {label!r}")
    try:
        return class_label_to_id(label)
    except KeyError:
        raise ValueError(f"Unknown hand class {label!r}") from None


def preflop_equity_vs_random(hand: str | Sequence[str | int], num_opponents: int = 1) -> float:
    """Preflop pot share against 1–9 uniform random hands, ties split equally.

    `hand` is a 169 class label ('AKs', 'TT', ' 72o') or two cards. Values for
    one and two opponents are exact; three or more come from 20M-sample Monte
    Carlo estimates (two-sided 95% Hoeffding radius ~0.03 pp).
    """
    num_opponents = _simulation_integer(num_opponents, "num_opponents", 1, PREFLOP_MAX_OPPONENTS)
    if isinstance(hand, str):
        class_id = _class_id(hand)
    else:
        cards = _simulation_cards(hand)
        if cards.size != 2 or cards[0] == cards[1]:
            raise ValueError("hand must be two distinct cards or a hand class label")
        class_id = class_label_to_id(combo_to_hand_class(cards))
    return float(_table(PREFLOP_VS_RANDOM_FILE)[class_id, num_opponents - 1])


def preflop_class_equities(classes: Sequence[str]) -> tuple[float, ...]:
    """Exact preflop equities of 2 or 3 hand classes, in the given order.

    Each value averages every legal deal of combos from those classes, with
    ties split equally. Three-way values are stored as float32.
    """
    if isinstance(classes, str) or len(classes) not in (2, 3):
        raise ValueError("classes must be a sequence of 2 or 3 hand class labels")
    ids = [_class_id(label) for label in classes]
    if len(ids) == 2:
        table = _table(PREFLOP_HU_FILE)
        return float(table[ids[0], ids[1]]), float(table[ids[1], ids[0]])

    order = sorted(range(3), key=ids.__getitem__)
    row = _table(PREFLOP_THREE_WAY_FILE)[packed_class_triple_index(*(ids[pos] for pos in order), _NUM_CLASSES)]
    if np.isnan(row[0]):
        raise ValueError(f"No legal deal for hand classes {tuple(classes)!r}")
    equities = [0.0] * 3
    for stored_pos, caller_pos in enumerate(order):
        equities[caller_pos] = float(row[stored_pos])
    return tuple(equities)


# Build helpers used by scripts/build_preflop_equity_tables.py.

def _heads_up_class_counts(class_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Twice the pot won (2*wins + ties) and runouts, summed over legal combo pairs of each class pair."""
    size = len(class_ids)
    num2 = np.zeros((size, size), dtype=np.int64)
    total = np.zeros((size, size), dtype=np.int64)
    combos = [tuple(class_combo_ids(int(class_id))) for class_id in class_ids]
    for first in range(size):
        for second in range(first, size):
            for matchup_key, multiplicity in _preflop_range_matchup_profile(combos[first], combos[second]):
                wins, ties, runouts = _preflop_canonical_counts(matchup_key)
                num2[first, second] += multiplicity * (2 * wins + ties)
                total[first, second] += multiplicity * runouts
            if second != first:
                num2[second, first] = 2 * total[first, second] - num2[first, second]
                total[second, first] = total[first, second]
    return num2, total


def _three_way_numerators(entries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Six times each seat's pot won, and runouts, for rows of 13 order counts."""
    counts = np.asarray(entries).astype(np.int64)
    return counts @ _THREE_WAY_WEIGHTS_X6, counts.sum(axis=1)


def _three_way_equity_rows(entries: np.ndarray) -> np.ndarray:
    num6, total = _three_way_numerators(entries)
    rows = np.full(num6.shape, np.nan)
    legal = total > 0
    rows[legal] = num6[legal] / (6.0 * total[legal, None])
    return rows


def _vs_two_random(seat_pots: np.ndarray, deals: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Pot won and deals of each class against two random hands; equity is their ratio.

    Rows follow sorted class triples a <= b <= c in packed order: `seat_pots`
    holds each seat's pot won and `deals` the weight of the row (both summed
    over legal combo triples). A hero of class x meets every ordered villain
    class pair, so each distinct class of a triple takes its seat once per
    ordering of the other two classes.
    """
    first, second, third = (np.array(axis) for axis in zip(*_sorted_triples(num_classes)))
    pot = np.zeros(num_classes, dtype=seat_pots.dtype)
    total = np.zeros(num_classes, dtype=deals.dtype)
    for pos, classes, counted, orderings in (
        (0, first, np.ones_like(first, dtype=bool), np.where(second == third, 1, 2)),
        (1, second, second != first, np.full_like(first, 2)),
        (2, third, third != second, np.where(first == second, 1, 2)),
    ):
        np.add.at(pot, classes[counted], orderings[counted] * seat_pots[counted, pos])
        np.add.at(total, classes[counted], orderings[counted] * deals[counted])
    return pot, total


def _legal_triple_counts(class_ids: Sequence[int]) -> np.ndarray:
    """Legal deals of one combo per seat for each sorted triple of `class_ids`, in packed order."""
    masks = canonical_combo_masks()
    legal = ((masks[:, None] & masks[None, :]) == 0).astype(np.int64)
    combos = [np.array(class_combo_ids(int(class_id))) for class_id in class_ids]
    membership = np.zeros((len(masks), len(combos)), dtype=np.int64)
    for local_id, combo_ids in enumerate(combos):
        membership[combo_ids, local_id] = 1
    counts = []
    for first in range(len(combos)):
        first_legal = legal[combos[first]]
        for second in range(first, len(combos)):
            pairs = legal[np.ix_(combos[first], combos[second])]
            # Legal (first, second) pairs that each third combo avoids, summed per class.
            per_combo = ((pairs @ legal[combos[second]]) * first_legal).sum(axis=0)
            counts.append((per_combo @ membership)[second:])
    return np.concatenate(counts)


def _sorted_triples(num_classes: int) -> list[tuple[int, int, int]]:
    return [
        (first, second, third)
        for first in range(num_classes)
        for second in range(first, num_classes)
        for third in range(second, num_classes)
    ]


def _expected_vs_random_runouts(class_id: int, num_opponents: int) -> int:
    villain_deals = 1
    for opponent in range(num_opponents):
        villain_deals *= comb(50 - 2 * opponent, 2)
    return len(class_combo_ids(int(class_id))) * villain_deals * comb(50 - 2 * num_opponents, 5)
