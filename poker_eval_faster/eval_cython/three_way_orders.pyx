# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, infer_types=True
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from array import array
from libc cimport stdint
import numpy as np

cimport cython
from poker_eval_faster.eval_cython.main cimport create_deck
from poker_eval_faster.eval_cython.main cimport handdat
from poker_eval_faster.eval_cython.common cimport fold_cards

cdef int EVAL_START = 53

cdef int ORDER_A_B_C = 0
cdef int ORDER_A_C_B = 1
cdef int ORDER_B_A_C = 2
cdef int ORDER_B_C_A = 3
cdef int ORDER_C_A_B = 4
cdef int ORDER_C_B_A = 5
cdef int ORDER_A_EQ_B_GT_C = 6
cdef int ORDER_A_EQ_C_GT_B = 7
cdef int ORDER_B_EQ_C_GT_A = 8
cdef int ORDER_A_GT_B_EQ_C = 9
cdef int ORDER_B_GT_A_EQ_C = 10
cdef int ORDER_C_GT_A_EQ_B = 11
cdef int ORDER_A_EQ_B_EQ_C = 12
cdef int NUM_THREE_WAY_ORDERS = 13


cdef inline void classify_three_way(stdint.uint32_t eval_a,
                                    stdint.uint32_t eval_b,
                                    stdint.uint32_t eval_c,
                                    stdint.uint64_t results[]) noexcept nogil:
    if eval_a == eval_b:
        if eval_b == eval_c:
            results[ORDER_A_EQ_B_EQ_C] += 1
        elif eval_a > eval_c:
            results[ORDER_A_EQ_B_GT_C] += 1
        else:
            results[ORDER_C_GT_A_EQ_B] += 1
        return

    if eval_a == eval_c:
        if eval_a > eval_b:
            results[ORDER_A_EQ_C_GT_B] += 1
        else:
            results[ORDER_B_GT_A_EQ_C] += 1
        return

    if eval_b == eval_c:
        if eval_b > eval_a:
            results[ORDER_B_EQ_C_GT_A] += 1
        else:
            results[ORDER_A_GT_B_EQ_C] += 1
        return

    if eval_a > eval_b:
        if eval_b > eval_c:
            results[ORDER_A_B_C] += 1
        elif eval_a > eval_c:
            results[ORDER_A_C_B] += 1
        else:
            results[ORDER_C_A_B] += 1
        return

    if eval_a > eval_c:
        results[ORDER_B_A_C] += 1
    elif eval_b > eval_c:
        results[ORDER_B_C_A] += 1
    else:
        results[ORDER_C_B_A] += 1


cdef void _enumerate_three_way_boards(int deck[], int len_deck,
                                      stdint.uint32_t sum_a,
                                      stdint.uint32_t sum_b,
                                      stdint.uint32_t sum_c,
                                      int len_board,
                                      stdint.uint64_t results[]) noexcept:
    cdef:
        int a, b, c, d, e
        stdint.uint32_t eval_a_a, eval_a_b, eval_a_c, eval_a_d, eval_a_e
        stdint.uint32_t eval_b_a, eval_b_b, eval_b_c, eval_b_d, eval_b_e
        stdint.uint32_t eval_c_a, eval_c_b, eval_c_c, eval_c_d, eval_c_e

    with nogil:
        for a in range(len_deck):
            eval_a_a = handdat[sum_a + deck[a]]
            eval_b_a = handdat[sum_b + deck[a]]
            eval_c_a = handdat[sum_c + deck[a]]
            if len_board == 4:
                classify_three_way(eval_a_a, eval_b_a, eval_c_a, results)
            elif len_board < 4:
                for b in range(a + 1, len_deck):
                    eval_a_b = handdat[eval_a_a + deck[b]]
                    eval_b_b = handdat[eval_b_a + deck[b]]
                    eval_c_b = handdat[eval_c_a + deck[b]]
                    if len_board == 3:
                        classify_three_way(eval_a_b, eval_b_b, eval_c_b, results)
                    elif len_board < 3:
                        for c in range(b + 1, len_deck):
                            eval_a_c = handdat[eval_a_b + deck[c]]
                            eval_b_c = handdat[eval_b_b + deck[c]]
                            eval_c_c = handdat[eval_c_b + deck[c]]
                            if len_board == 2:
                                classify_three_way(eval_a_c, eval_b_c, eval_c_c, results)
                            elif len_board < 2:
                                for d in range(c + 1, len_deck):
                                    eval_a_d = handdat[eval_a_c + deck[d]]
                                    eval_b_d = handdat[eval_b_c + deck[d]]
                                    eval_c_d = handdat[eval_c_c + deck[d]]
                                    if len_board == 1:
                                        classify_three_way(eval_a_d, eval_b_d, eval_c_d, results)
                                    elif len_board < 1:
                                        for e in range(d + 1, len_deck):
                                            eval_a_e = handdat[eval_a_d + deck[e]]
                                            eval_b_e = handdat[eval_b_d + deck[e]]
                                            eval_c_e = handdat[eval_c_d + deck[e]]
                                            classify_three_way(eval_a_e, eval_b_e, eval_c_e, results)


cdef void _evaluate_three_way_orders(int hand_a1, int hand_a2,
                                     int hand_b1, int hand_b2,
                                     int hand_c1, int hand_c2,
                                     int[:] board,
                                     stdint.uint64_t results[]):
    cdef:
        int len_board = board.size
        int len_total = len_board + 6
        stdint.uint32_t sum_board
        stdint.uint32_t sum_a
        stdint.uint32_t sum_b
        stdint.uint32_t sum_c
        int *dead_cards = <int *>PyMem_Malloc(len_total * sizeof(int))
        int *deck = <int *>PyMem_Malloc((52 - len_total) * sizeof(int))
        int i

    dead_cards[0] = hand_a1
    dead_cards[1] = hand_a2
    dead_cards[2] = hand_b1
    dead_cards[3] = hand_b2
    dead_cards[4] = hand_c1
    dead_cards[5] = hand_c2
    for i in range(len_board):
        dead_cards[6 + i] = board[i]

    cdef int[:] dead_cards_mv = <int[:len_total]> dead_cards
    cdef int len_deck = create_deck(dead_cards_mv, len_total, deck)
    PyMem_Free(dead_cards)

    sum_board = fold_cards(EVAL_START, board, 0, len_board)
    sum_a = handdat[handdat[sum_board + hand_a1] + hand_a2]
    sum_b = handdat[handdat[sum_board + hand_b1] + hand_b2]
    sum_c = handdat[handdat[sum_board + hand_c1] + hand_c2]

    if len_board < 5:
        _enumerate_three_way_boards(deck, len_deck, sum_a, sum_b, sum_c, len_board, results)
    else:
        with nogil:
            classify_three_way(sum_a, sum_b, sum_c, results)

    PyMem_Free(deck)


cpdef object evaluate_three_way_orders_c(int[:] hands, int[:] board=array('i', [])):
    if hands.size != 6:
        raise ValueError(f"Three-way evaluation expects exactly 6 cards, got {hands.size}")

    cdef:
        object results_np = np.zeros(NUM_THREE_WAY_ORDERS, dtype=np.uint64)
        stdint.uint64_t[::1] results = results_np

    _evaluate_three_way_orders(
        hands[0], hands[1],
        hands[2], hands[3],
        hands[4], hands[5],
        board,
        &results[0],
    )
    return results_np
