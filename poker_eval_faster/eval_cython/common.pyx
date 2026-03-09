# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, infer_types=True
from libc cimport stdint

from poker_eval_faster.eval_cython.main cimport handdat


cdef stdint.uint32_t fold_cards(stdint.uint32_t start_eval, int[:] cards, int start_idx, int end_idx) noexcept nogil:
    cdef stdint.uint32_t p = start_eval
    cdef int i
    for i in range(start_idx, end_idx):
        p = handdat[p + cards[i]]
    return p
