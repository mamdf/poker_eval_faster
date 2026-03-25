from libc cimport stdint


cdef void _evaluate_three_way_orders(int hand_a1, int hand_a2,
                                     int hand_b1, int hand_b2,
                                     int hand_c1, int hand_c2,
                                     int[:] board,
                                     stdint.uint64_t results[])
