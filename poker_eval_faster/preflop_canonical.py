from functools import lru_cache
from itertools import permutations
from typing import Sequence, Tuple

import numpy as np

from .eval_cython.hands_evaluate import evaluate_heads_up_counts_c


_EMPTY_BOARD = np.array([], dtype="int32")
_NUM_SUITS = 4
_SUIT_PERMUTATIONS = tuple(permutations(range(_NUM_SUITS)))


def _cards_mask(cards: Sequence[int]) -> np.uint64:
    mask = np.uint64(0)
    for card in cards:
        mask |= np.uint64(1) << np.uint64(int(card) - 1)
    return mask


@lru_cache(maxsize=1)
def canonical_combos() -> np.ndarray:
    return np.array([(a, b) for a in range(1, 53) for b in range(a + 1, 53)], dtype="int32")


@lru_cache(maxsize=1)
def canonical_combo_masks() -> np.ndarray:
    combos = canonical_combos()
    masks = np.zeros(combos.shape[0], dtype=np.uint64)
    for idx, (first, second) in enumerate(combos):
        masks[idx] = _cards_mask((int(first), int(second)))
    return masks


@lru_cache(maxsize=1)
def _combo_index_map() -> dict[Tuple[int, int], int]:
    combos = canonical_combos()
    return {
        (int(first), int(second)): idx
        for idx, (first, second) in enumerate(combos)
    }


def _combo_ids_from_array(combo_arr: np.ndarray) -> Tuple[int, ...]:
    index_map = _combo_index_map()
    combo_ids = []
    for first, second in combo_arr.tolist():
        first_i = int(first)
        second_i = int(second)
        if first_i > second_i:
            first_i, second_i = second_i, first_i
        combo_ids.append(index_map[(first_i, second_i)])
    return tuple(combo_ids)


def _remap_card_suit(card: int, suit_permutation: Tuple[int, int, int, int]) -> int:
    zero_based = card - 1
    rank = zero_based // _NUM_SUITS
    suit = zero_based % _NUM_SUITS
    return (rank * _NUM_SUITS) + suit_permutation[suit] + 1


@lru_cache(maxsize=200000)
def _canonical_preflop_matchup(hero_combo_idx: int, villain_combo_idx: int) -> Tuple[int, int, int, int]:
    # Preflop equities are invariant under global suit renaming, so we cache the
    # lexicographically smallest representative across the 24 suit permutations.
    combos = canonical_combos()
    hero_first, hero_second = combos[int(hero_combo_idx)]
    villain_first, villain_second = combos[int(villain_combo_idx)]
    best_key: Tuple[int, int, int, int] | None = None

    for suit_permutation in _SUIT_PERMUTATIONS:
        hero_cards = sorted((
            _remap_card_suit(int(hero_first), suit_permutation),
            _remap_card_suit(int(hero_second), suit_permutation),
        ))
        villain_cards = sorted((
            _remap_card_suit(int(villain_first), suit_permutation),
            _remap_card_suit(int(villain_second), suit_permutation),
        ))
        candidate = (
            hero_cards[0],
            hero_cards[1],
            villain_cards[0],
            villain_cards[1],
        )
        if best_key is None or candidate < best_key:
            best_key = candidate

    return best_key if best_key is not None else (0, 0, 0, 0)


@lru_cache(maxsize=50000)
def _preflop_canonical_counts(matchup_key: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    hero = np.array(matchup_key[:2], dtype="int32")
    villain = np.array(matchup_key[2:], dtype="int32")
    counts = evaluate_heads_up_counts_c(hero, villain, _EMPTY_BOARD)
    return int(counts[0]), int(counts[1]), int(counts[2])


@lru_cache(maxsize=200000)
def _preflop_combo_pair_counts(combo_a_idx: int, combo_b_idx: int) -> Tuple[int, int, int]:
    if combo_a_idx == combo_b_idx:
        return 0, 0, 0
    if combo_a_idx > combo_b_idx:
        combo_a_idx, combo_b_idx = combo_b_idx, combo_a_idx

    masks = canonical_combo_masks()
    if int(masks[combo_a_idx]) & int(masks[combo_b_idx]):
        return 0, 0, 0

    matchup_key = _canonical_preflop_matchup(combo_a_idx, combo_b_idx)
    return _preflop_canonical_counts(matchup_key)


@lru_cache(maxsize=256)
def _preflop_range_matchup_profile(hero_combo_ids: Tuple[int, ...], villain_combo_ids: Tuple[int, ...]) -> Tuple[Tuple[Tuple[int, int, int, int], int], ...]:
    # Group repeated combo-vs-combo collisions so range aggregation only touches
    # each exact preflop matchup class once.
    masks = canonical_combo_masks()
    grouped: dict[Tuple[int, int, int, int], int] = {}

    for hero_idx in hero_combo_ids:
        hero_mask = int(masks[int(hero_idx)])
        for villain_idx in villain_combo_ids:
            if hero_mask & int(masks[int(villain_idx)]):
                continue
            matchup_key = _canonical_preflop_matchup(int(hero_idx), int(villain_idx))
            grouped[matchup_key] = grouped.get(matchup_key, 0) + 1

    return tuple(grouped.items())


def _evaluate_ranges_preflop_cached(hero_combo_ids: Tuple[int, ...], villain_combo_ids: Tuple[int, ...]) -> float:
    total_hero = 0.0
    total_all = 0

    for matchup_key, multiplicity in _preflop_range_matchup_profile(hero_combo_ids, villain_combo_ids):
        wins, ties, total = _preflop_canonical_counts(matchup_key)
        total_hero += multiplicity * (wins + (ties / 2.0))
        total_all += multiplicity * total

    if total_all == 0:
        return 0.0
    return total_hero / total_all


def _clear_preflop_canonical_caches() -> None:
    _canonical_preflop_matchup.cache_clear()
    _preflop_canonical_counts.cache_clear()
    _preflop_combo_pair_counts.cache_clear()
    _preflop_range_matchup_profile.cache_clear()
