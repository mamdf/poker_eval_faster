from functools import lru_cache
from itertools import permutations
from typing import Sequence, Tuple

import numpy as np

from .eval_cython.hands_evaluate import evaluate_heads_up_counts_c
from .eval_cython.three_way_orders import evaluate_three_way_orders_c


_EMPTY_BOARD = np.array([], dtype="int32")
_NUM_SUITS = 4
_NUM_THREE_WAY_ORDERS = 13
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


@lru_cache(maxsize=1)
def _combo_suit_remaps() -> np.ndarray:
    combos = canonical_combos()
    remaps = np.zeros((len(_SUIT_PERMUTATIONS), combos.shape[0], 2), dtype=np.uint8)

    for perm_idx, suit_permutation in enumerate(_SUIT_PERMUTATIONS):
        for combo_idx, (first, second) in enumerate(combos):
            remapped_first = _remap_card_suit(int(first), suit_permutation)
            remapped_second = _remap_card_suit(int(second), suit_permutation)
            if remapped_first < remapped_second:
                remaps[perm_idx, combo_idx, 0] = remapped_first
                remaps[perm_idx, combo_idx, 1] = remapped_second
            else:
                remaps[perm_idx, combo_idx, 0] = remapped_second
                remaps[perm_idx, combo_idx, 1] = remapped_first

    return remaps


@lru_cache(maxsize=300000)
def _canonical_preflop_combo_tuple(combo_indices: Tuple[int, ...]) -> Tuple[int, ...]:
    # Preflop equities are invariant under global suit renaming, so we cache the
    # lexicographically smallest representative across the 24 suit permutations.
    remaps = _combo_suit_remaps()
    best_key: Tuple[int, ...] | None = None

    for perm_idx in range(remaps.shape[0]):
        candidate_cards: list[int] = []
        for combo_idx in combo_indices:
            cards = remaps[perm_idx, int(combo_idx)]
            candidate_cards.extend((int(cards[0]), int(cards[1])))
        candidate = tuple(candidate_cards)
        if best_key is None or candidate < best_key:
            best_key = candidate

    return best_key if best_key is not None else tuple()


@lru_cache(maxsize=200000)
def _canonical_preflop_matchup(hero_combo_idx: int, villain_combo_idx: int) -> Tuple[int, int, int, int]:
    return _canonical_preflop_combo_tuple((hero_combo_idx, villain_combo_idx))


@lru_cache(maxsize=300000)
def _canonical_preflop_three_way_matchup(combo_a_idx: int, combo_b_idx: int, combo_c_idx: int) -> Tuple[int, int, int, int, int, int]:
    return _canonical_preflop_combo_tuple((combo_a_idx, combo_b_idx, combo_c_idx))


@lru_cache(maxsize=50000)
def _preflop_canonical_counts(matchup_key: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    hero = np.array(matchup_key[:2], dtype="int32")
    villain = np.array(matchup_key[2:], dtype="int32")
    counts = evaluate_heads_up_counts_c(hero, villain, _EMPTY_BOARD)
    return int(counts[0]), int(counts[1]), int(counts[2])


@lru_cache(maxsize=100000)
def _preflop_three_way_canonical_counts(matchup_key: Tuple[int, int, int, int, int, int]) -> Tuple[int, ...]:
    hands = np.array(matchup_key, dtype="int32")
    counts = evaluate_three_way_orders_c(hands, _EMPTY_BOARD)
    return tuple(int(value) for value in counts.tolist())


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


@lru_cache(maxsize=300000)
def _preflop_three_way_combo_order_counts(combo_a_idx: int, combo_b_idx: int, combo_c_idx: int) -> Tuple[int, ...]:
    masks = canonical_combo_masks()
    mask_a = int(masks[int(combo_a_idx)])
    mask_b = int(masks[int(combo_b_idx)])
    mask_c = int(masks[int(combo_c_idx)])
    if mask_a & mask_b or mask_a & mask_c or mask_b & mask_c:
        return (0,) * _NUM_THREE_WAY_ORDERS

    matchup_key = _canonical_preflop_three_way_matchup(combo_a_idx, combo_b_idx, combo_c_idx)
    return _preflop_three_way_canonical_counts(matchup_key)


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


@lru_cache(maxsize=128)
def _preflop_three_way_range_matchup_profile(
    range_a_combo_ids: Tuple[int, ...],
    range_b_combo_ids: Tuple[int, ...],
    range_c_combo_ids: Tuple[int, ...],
) -> Tuple[Tuple[Tuple[int, int, int, int, int, int], int], ...]:
    masks = canonical_combo_masks()
    grouped: dict[Tuple[int, int, int, int, int, int], int] = {}

    for combo_a_idx in range_a_combo_ids:
        mask_a = int(masks[int(combo_a_idx)])
        for combo_b_idx in range_b_combo_ids:
            mask_b = int(masks[int(combo_b_idx)])
            if mask_a & mask_b:
                continue
            mask_ab = mask_a | mask_b
            for combo_c_idx in range_c_combo_ids:
                if mask_ab & int(masks[int(combo_c_idx)]):
                    continue
                matchup_key = _canonical_preflop_three_way_matchup(
                    int(combo_a_idx),
                    int(combo_b_idx),
                    int(combo_c_idx),
                )
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


@lru_cache(maxsize=128)
def _evaluate_three_way_ranges_preflop_cached(
    range_a_combo_ids: Tuple[int, ...],
    range_b_combo_ids: Tuple[int, ...],
    range_c_combo_ids: Tuple[int, ...],
) -> Tuple[int, ...]:
    total_counts = [0] * _NUM_THREE_WAY_ORDERS

    for matchup_key, multiplicity in _preflop_three_way_range_matchup_profile(
        range_a_combo_ids,
        range_b_combo_ids,
        range_c_combo_ids,
    ):
        counts = _preflop_three_way_canonical_counts(matchup_key)
        for idx, count in enumerate(counts):
            if count == 0:
                continue
            total_counts[idx] += multiplicity * count

    return tuple(total_counts)


def _clear_preflop_canonical_caches() -> None:
    _canonical_preflop_combo_tuple.cache_clear()
    _canonical_preflop_matchup.cache_clear()
    _canonical_preflop_three_way_matchup.cache_clear()
    _preflop_canonical_counts.cache_clear()
    _preflop_three_way_canonical_counts.cache_clear()
    _preflop_combo_pair_counts.cache_clear()
    _preflop_three_way_combo_order_counts.cache_clear()
    _preflop_range_matchup_profile.cache_clear()
    _preflop_three_way_range_matchup_profile.cache_clear()
    _evaluate_three_way_ranges_preflop_cached.cache_clear()
