from libc cimport stdint

cdef int create_deck(int[:] dead_cards, int len_dead_cards, int results[])
cdef const stdint.uint32_t[:] handdat
