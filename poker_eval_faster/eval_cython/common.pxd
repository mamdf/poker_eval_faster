from libc cimport stdint


cdef stdint.uint32_t fold_cards(stdint.uint32_t start_eval, int[:] cards, int start_idx, int end_idx) noexcept nogil
