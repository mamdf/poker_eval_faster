# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, infer_types=True
from array import array
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc cimport stdint
from libc.stddef cimport size_t
from libc.stdlib cimport qsort
import numpy as np

from poker_eval_faster.eval_cython.three_way_orders cimport _evaluate_three_way_orders
from poker_eval_faster.preflop_canonical import _combo_suit_remaps, canonical_combo_masks


cdef int NUM_THREE_WAY_ORDERS = 13
cdef int NUM_SUIT_PERMUTATIONS = 24
cdef int MAX_CANONICAL_COUNT_CACHE_SIZE = 100000

cdef object _EMPTY_BOARD = array('i', [])
cdef object _COMBO_MASKS_OBJ = None
cdef object _COMBO_SUIT_REMAPS_OBJ = None
cdef dict _CANONICAL_COUNT_CACHE = {}


cdef inline void _ensure_lookup_tables():
    global _COMBO_MASKS_OBJ, _COMBO_SUIT_REMAPS_OBJ
    if _COMBO_MASKS_OBJ is None:
        _COMBO_MASKS_OBJ = canonical_combo_masks()
    if _COMBO_SUIT_REMAPS_OBJ is None:
        _COMBO_SUIT_REMAPS_OBJ = _combo_suit_remaps()


cdef inline stdint.uint64_t _pack_matchup_key(unsigned char a1,
                                              unsigned char a2,
                                              unsigned char b1,
                                              unsigned char b2,
                                              unsigned char c1,
                                              unsigned char c2) noexcept nogil:
    return (
        (<stdint.uint64_t>a1 << 40)
        | (<stdint.uint64_t>a2 << 32)
        | (<stdint.uint64_t>b1 << 24)
        | (<stdint.uint64_t>b2 << 16)
        | (<stdint.uint64_t>c1 << 8)
        | <stdint.uint64_t>c2
    )


cdef inline stdint.uint64_t _canonical_three_way_matchup_key(
    int combo_a_idx,
    int combo_b_idx,
    int combo_c_idx,
    unsigned char[:, :, :] combo_remaps,
) noexcept nogil:
    cdef int perm_idx
    cdef stdint.uint64_t best_key = 0
    cdef stdint.uint64_t candidate

    for perm_idx in range(NUM_SUIT_PERMUTATIONS):
        candidate = _pack_matchup_key(
            combo_remaps[perm_idx, combo_a_idx, 0],
            combo_remaps[perm_idx, combo_a_idx, 1],
            combo_remaps[perm_idx, combo_b_idx, 0],
            combo_remaps[perm_idx, combo_b_idx, 1],
            combo_remaps[perm_idx, combo_c_idx, 0],
            combo_remaps[perm_idx, combo_c_idx, 1],
        )
        if perm_idx == 0 or candidate < best_key:
            best_key = candidate
    return best_key


cdef int _compare_uint64(const void *left, const void *right) noexcept nogil:
    cdef stdint.uint64_t left_value = (<stdint.uint64_t *> left)[0]
    cdef stdint.uint64_t right_value = (<stdint.uint64_t *> right)[0]
    if left_value < right_value:
        return -1
    if left_value > right_value:
        return 1
    return 0


cpdef object evaluate_three_way_class_counts_c(int[:] class_a_combo_ids,
                                               int[:] class_b_combo_ids,
                                               int[:] class_c_combo_ids):
    _ensure_lookup_tables()

    cdef:
        stdint.uint64_t[:] combo_masks = _COMBO_MASKS_OBJ
        unsigned char[:, :, :] combo_remaps = _COMBO_SUIT_REMAPS_OBJ
        int[:] empty_board = _EMPTY_BOARD
        int combo_a_idx
        int combo_b_idx
        int combo_c_idx
        int len_a = class_a_combo_ids.shape[0]
        int len_b = class_b_combo_ids.shape[0]
        int len_c = class_c_combo_ids.shape[0]
        Py_ssize_t max_matchups = len_a * len_b * len_c
        Py_ssize_t matchup_count = 0
        Py_ssize_t matchup_idx
        stdint.uint64_t mask_a
        stdint.uint64_t mask_b
        stdint.uint64_t mask_c
        stdint.uint64_t mask_ab
        stdint.uint64_t current_key
        stdint.uint64_t multiplicity
        stdint.uint64_t matchup_keys_tmp[1]
        stdint.uint64_t *matchup_keys = matchup_keys_tmp
        stdint.uint64_t order_counts_buffer[13]
        object cached_counts
        list cached_counts_list
        int cached_count
        object results_np = np.zeros(NUM_THREE_WAY_ORDERS, dtype=np.uint64)
        stdint.uint64_t[::1] results = results_np
        int order_idx

    if max_matchups > 0:
        matchup_keys = <stdint.uint64_t *>PyMem_Malloc(max_matchups * sizeof(stdint.uint64_t))
        if matchup_keys == NULL:
            raise MemoryError()

    try:
        for combo_a_idx in class_a_combo_ids:
            mask_a = combo_masks[combo_a_idx]
            for combo_b_idx in class_b_combo_ids:
                mask_b = combo_masks[combo_b_idx]
                if mask_a & mask_b:
                    continue
                mask_ab = mask_a | mask_b
                for combo_c_idx in class_c_combo_ids:
                    mask_c = combo_masks[combo_c_idx]
                    if mask_ab & mask_c:
                        continue
                    matchup_keys[matchup_count] = _canonical_three_way_matchup_key(
                        combo_a_idx,
                        combo_b_idx,
                        combo_c_idx,
                        combo_remaps,
                    )
                    matchup_count += 1

        if matchup_count == 0:
            return results_np

        qsort(matchup_keys, <size_t>matchup_count, sizeof(stdint.uint64_t), _compare_uint64)

        current_key = matchup_keys[0]
        multiplicity = 1
        for matchup_idx in range(1, matchup_count + 1):
            if matchup_idx < matchup_count and matchup_keys[matchup_idx] == current_key:
                multiplicity += 1
                continue

            cached_counts = _CANONICAL_COUNT_CACHE.get(current_key)
            if cached_counts is None:
                for order_idx in range(NUM_THREE_WAY_ORDERS):
                    order_counts_buffer[order_idx] = 0

                _evaluate_three_way_orders(
                    <int>((current_key >> 40) & 0xFF),
                    <int>((current_key >> 32) & 0xFF),
                    <int>((current_key >> 24) & 0xFF),
                    <int>((current_key >> 16) & 0xFF),
                    <int>((current_key >> 8) & 0xFF),
                    <int>(current_key & 0xFF),
                    empty_board,
                    order_counts_buffer,
                )
                cached_counts_list = []
                for order_idx in range(NUM_THREE_WAY_ORDERS):
                    cached_counts_list.append(int(order_counts_buffer[order_idx]))
                cached_counts = tuple(cached_counts_list)
                if len(_CANONICAL_COUNT_CACHE) >= MAX_CANONICAL_COUNT_CACHE_SIZE:
                    _CANONICAL_COUNT_CACHE.clear()
                _CANONICAL_COUNT_CACHE[current_key] = cached_counts

            for order_idx in range(NUM_THREE_WAY_ORDERS):
                cached_count = cached_counts[order_idx]
                if cached_count != 0:
                    results[order_idx] += multiplicity * cached_count

            if matchup_idx < matchup_count:
                current_key = matchup_keys[matchup_idx]
                multiplicity = 1

        return results_np
    finally:
        if max_matchups > 0 and matchup_keys != matchup_keys_tmp:
            PyMem_Free(matchup_keys)


cpdef clear_three_way_class_builder_cache_c():
    _CANONICAL_COUNT_CACHE.clear()
