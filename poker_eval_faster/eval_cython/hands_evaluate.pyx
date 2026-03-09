# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, infer_types=True
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from array import array
from libc cimport stdint
import numpy as np

cimport cython
from poker_eval_faster.eval_cython.main cimport handdat
from poker_eval_faster.eval_cython.main cimport create_deck
from poker_eval_faster.eval_cython.common cimport fold_cards

cdef int EVAL_START = 53


cdef inline void sum_new_card(int new_card, stdint.uint32_t sum_hands[], int num_hands, stdint.uint32_t new_sum_hands[]) noexcept nogil:
    """
    :param new_card: card rank (1,53) to sum to each eval hand value
    :param sum_hands: array with eval hands values
    :param num_hands: sum hands length
    :param new_sum_hands: array to save new eval hands values
    :return: None
    """
    cdef int i
    for i in range(num_hands):
        new_sum_hands[i] = handdat[sum_hands[i] + new_card]


cdef inline void eval_heads_up(stdint.uint32_t hero_eval,
                               stdint.uint32_t villain_eval,
                               stdint.uint64_t results[]) noexcept nogil:
    if hero_eval > villain_eval:
        results[0] += 1
    elif hero_eval == villain_eval:
        results[1] += 1
    results[2] += 1

@cython.cdivision(True)
cdef inline void eval_hands(stdint.uint32_t sum_hands[], int num_hands, double results[]) noexcept nogil:
    """
    Compare two or more eval hands numbers and figured out winner (or ties)
    :param sum_hands: array with eval hands values
    :param num_hands: sum hands length
    :param results: array to save number of win and tie per hand
    :return: None
    """
    cdef int i, win_id = 0
    cdef double tie = 0.0
    cdef stdint.uint32_t max_eval = 0
    cdef bint possible_tie = False
    for i in range(num_hands):
        if sum_hands[i] > max_eval:
            max_eval = sum_hands[i]
            win_id = i
        elif sum_hands[i] == max_eval:
            possible_tie = True

    if possible_tie:
        for i in range(num_hands):
            if sum_hands[i] == max_eval:
                tie += 1

    if tie > 1:
        for i in range(num_hands):
            if sum_hands[i] == max_eval:
                results[num_hands + i] += 1 / tie
    else:
        results[win_id] += 1


@cython.cdivision(True)
cpdef list[float] hands_to_equity(double[:] results):
    cdef:
        double total = 0.0
        int i
        int len_results = len(results)
        int len_equity = len_results // 2
    equity = [0.0] * len_equity
    for i in range(len_results):
        total += results[i]
    for i in range(len_equity):
        equity[i] = (results[i] + results[i + len_equity]) / total
    return equity


cdef void _create_boards(int deck[], int len_deck, stdint.uint32_t sum_hands[],
                         int num_hands, int len_board,
                         double results[]):
    """
    combinations throughout twoplustwo array, create all combinations of board cards (5 - len_board).
    Evaluate hands and save results array [win, tie]
    """
    cdef:
        int a, b, c, d, e
        stdint.uint32_t *scratch = <stdint.uint32_t *>PyMem_Malloc(5 * num_hands * sizeof(stdint.uint32_t))
        stdint.uint32_t *new_sum_hands_a = scratch
        stdint.uint32_t *new_sum_hands_b = scratch + num_hands
        stdint.uint32_t *new_sum_hands_c = scratch + (2 * num_hands)
        stdint.uint32_t *new_sum_hands_d = scratch + (3 * num_hands)
        stdint.uint32_t *new_sum_hands_e = scratch + (4 * num_hands)

    if len_board < 5:
        with nogil:
            for a in range(len_deck):
                sum_new_card(deck[a], sum_hands, num_hands, new_sum_hands_a)
                if len_board == 4:
                    eval_hands(new_sum_hands_a, num_hands, results)
                if len_board < 4:
                    for b in range(a+1, len_deck):
                        sum_new_card(deck[b], new_sum_hands_a, num_hands, new_sum_hands_b)
                        if len_board == 3:
                            eval_hands(new_sum_hands_b, num_hands, results)
                        if len_board < 3:
                            for c in range(b+1, len_deck):
                                sum_new_card(deck[c], new_sum_hands_b, num_hands, new_sum_hands_c)
                                for d in range(c+1, len_deck):
                                    sum_new_card(deck[d], new_sum_hands_c, num_hands, new_sum_hands_d)
                                    for e in range(d+1, len_deck):
                                        sum_new_card(deck[e], new_sum_hands_d, num_hands, new_sum_hands_e)
                                        eval_hands(new_sum_hands_e, num_hands, results)

    PyMem_Free(scratch)
    return


cdef void _create_heads_up_boards(int deck[], int len_deck,
                                  stdint.uint32_t hero_sum,
                                  stdint.uint32_t villain_sum,
                                  int len_board,
                                  stdint.uint64_t results[]) noexcept:
    cdef:
        int a, b, c, d, e
        stdint.uint32_t hero_a, hero_b, hero_c, hero_d, hero_e
        stdint.uint32_t villain_a, villain_b, villain_c, villain_d, villain_e

    with nogil:
        for a in range(len_deck):
            hero_a = handdat[hero_sum + deck[a]]
            villain_a = handdat[villain_sum + deck[a]]
            if len_board == 4:
                eval_heads_up(hero_a, villain_a, results)
            elif len_board < 4:
                for b in range(a + 1, len_deck):
                    hero_b = handdat[hero_a + deck[b]]
                    villain_b = handdat[villain_a + deck[b]]
                    if len_board == 3:
                        eval_heads_up(hero_b, villain_b, results)
                    elif len_board < 3:
                        for c in range(b + 1, len_deck):
                            hero_c = handdat[hero_b + deck[c]]
                            villain_c = handdat[villain_b + deck[c]]
                            if len_board == 2:
                                eval_heads_up(hero_c, villain_c, results)
                            elif len_board < 2:
                                for d in range(c + 1, len_deck):
                                    hero_d = handdat[hero_c + deck[d]]
                                    villain_d = handdat[villain_c + deck[d]]
                                    if len_board == 1:
                                        eval_heads_up(hero_d, villain_d, results)
                                    elif len_board < 1:
                                        for e in range(d + 1, len_deck):
                                            hero_e = handdat[hero_d + deck[e]]
                                            villain_e = handdat[villain_d + deck[e]]
                                            eval_heads_up(hero_e, villain_e, results)


cdef void _evaluate_heads_up_counts(int hero1, int hero2,
                                    int villain1, int villain2,
                                    int[:] board,
                                    stdint.uint64_t results[]):
    cdef:
        int len_board = board.size
        int len_total = len_board + 4
        stdint.uint32_t sum_board
        stdint.uint32_t hero_sum
        stdint.uint32_t villain_sum
        int *dead_cards = <int *>PyMem_Malloc(len_total * sizeof(int))
        int *deck = <int *>PyMem_Malloc((52 - len_total) * sizeof(int))
        int i

    dead_cards[0] = hero1
    dead_cards[1] = hero2
    dead_cards[2] = villain1
    dead_cards[3] = villain2
    for i in range(len_board):
        dead_cards[4 + i] = board[i]

    cdef int[:] dead_cards_mv = <int[:len_total]> dead_cards
    cdef int len_deck = create_deck(dead_cards_mv, len_total, deck)
    PyMem_Free(dead_cards)

    sum_board = fold_cards(EVAL_START, board, 0, len_board)
    hero_sum = handdat[handdat[sum_board + hero1] + hero2]
    villain_sum = handdat[handdat[sum_board + villain1] + villain2]

    if len_board < 5:
        _create_heads_up_boards(deck, len_deck, hero_sum, villain_sum, len_board, results)
    else:
        with nogil:
            eval_heads_up(hero_sum, villain_sum, results)

    PyMem_Free(deck)


cpdef object evaluate_hands_c(int[:] hands, int[:] board=array('i', [])):
    """
    Evaluate hands vs Board, if board is incomplete (< 5) complete it with all possible cards and eval it.
    :return: [win hand 1, win hand 2, tie hand 1, tie hand 2 ... ]
    """
    cdef:
        stdint.uint32_t tmp_sum, sum_board
        int len_hands = hands.size
        int len_board = board.size
        int num_hands = len_hands // 2
        int len_total = len_board + len_hands
        stdint.uint32_t *sum_hands = <stdint.uint32_t *>PyMem_Malloc(num_hands * sizeof(stdint.uint32_t))
        int *deck = <int *>PyMem_Malloc((52 - len_total) * sizeof(int))
        int *dead_cards = <int *>PyMem_Malloc(len_total * sizeof(int))
        # results gestionado por Python para evitar fugas al devolverlo
        object results_np = np.zeros(len_hands, dtype=np.float64)
        cdef double[::1] results = results_np
        int i
    # hands and board to one array dead cards
    for i in range(len_hands):
        dead_cards[i] = hands[i]
    for i in range(len_board):
        dead_cards[len_hands + i] = board[i]
    cdef int[:] dead_cards_py = <int[:len_total]> dead_cards
    cdef int len_deck = create_deck(dead_cards_py, len_total, deck)  # create deck less dead cards
    PyMem_Free(dead_cards)
    # eval board
    sum_board = fold_cards(EVAL_START, board, 0, len_board)
    # eval hands
    for i in range(num_hands):
        # sumar dos cartas de la mano sobre el board
        sum_hands[i] = handdat[handdat[sum_board + hands[i * 2]] + hands[i * 2 + 1]]
    # brute force fill board with all cards and eval it
    if len_board < 5:
        _create_boards(deck, len_deck, sum_hands, num_hands, len_board, &results[0])
    else:  # eval board, each hand in results
        eval_hands(sum_hands, num_hands, &results[0])

    PyMem_Free(sum_hands)
    PyMem_Free(deck)

    return results_np


cpdef object evaluate_heads_up_counts_c(int[:] hero, int[:] villain, int[:] board=array('i', [])):
    cdef:
        object results_np = np.zeros(3, dtype=np.uint64)
        stdint.uint64_t[::1] results = results_np
    _evaluate_heads_up_counts(hero[0], hero[1], villain[0], villain[1], board, &results[0])
    return results_np


@cython.cdivision(True)
cpdef double evaluate_range_vs_range_c(int[:,:] hero_combos, int[:,:] villain_combos, int[:] board=array('i', [])):
    """
    Evalúa equity agregada de un rango (lista de combos) vs otro rango.
    Cada combo es un par de enteros (carta1, carta2) en hero_combos/villain_combos.
    Devuelve la equity promedio del rango héroe contra el rango villano.
    """
    cdef:
        Py_ssize_t i, j, k
        int len_board = board.size
        int h1, h2, v1, v2
        double total_hero = 0.0
        double total_all = 0.0
        bint ok
        stdint.uint64_t counts[3]
    for i in range(hero_combos.shape[0]):
        h1 = hero_combos[i, 0]
        h2 = hero_combos[i, 1]
        if h1 == h2:
            continue
        for j in range(villain_combos.shape[0]):
            v1 = villain_combos[j, 0]
            v2 = villain_combos[j, 1]
            # validar cartas distintas entre sí y con la otra mano
            if v1 == v2:
                continue
            if h1 == v1 or h1 == v2 or h2 == v1 or h2 == v2:
                continue
            # validar contra el board
            ok = True
            for k in range(len_board):
                if board[k] == h1 or board[k] == h2 or board[k] == v1 or board[k] == v2:
                    ok = False
                    break
            if not ok:
                continue
            counts[0] = 0
            counts[1] = 0
            counts[2] = 0
            _evaluate_heads_up_counts(h1, h2, v1, v2, board, counts)
            total_hero += counts[0] + (counts[1] / 2.0)
            total_all += counts[2]

    if total_all == 0.0:
        return 0.0
    return total_hero / total_all
