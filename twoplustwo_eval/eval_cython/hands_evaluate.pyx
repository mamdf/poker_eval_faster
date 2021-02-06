from cpython.mem cimport PyMem_Malloc, PyMem_Free
from array import array
from libc cimport stdint

cimport cython
from twoplustwo_eval.eval_cython.main cimport handdat
from twoplustwo_eval.eval_cython.main cimport create_deck


cdef inline void sum_new_card(int new_card, stdint.uint32_t sum_hands[], int num_hands, stdint.uint32_t new_sum_hands[]):
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

@cython.cdivision(True)
cdef inline void eval_hands(stdint.uint32_t sum_hands[], int num_hands, double results[]):
    """
    Compare two or more eval hands numbers and figured out winner (or ties)
    :param sum_hands: array with eval hands values
    :param num_hands: sum hands length
    :param results: array to save number of win and tie per hand
    :return: None
    """
    cdef int i, win_id
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
        for i in range(num_hands):  # count tie only for win hands
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
        stdint.uint32_t *new_sum_hands_a = <stdint.uint32_t *>PyMem_Malloc(num_hands * sizeof(stdint.uint32_t))
        stdint.uint32_t *new_sum_hands_b = <stdint.uint32_t *>PyMem_Malloc(num_hands * sizeof(stdint.uint32_t))
        stdint.uint32_t *new_sum_hands_c = <stdint.uint32_t *>PyMem_Malloc(num_hands * sizeof(stdint.uint32_t))
        stdint.uint32_t *new_sum_hands_d = <stdint.uint32_t *>PyMem_Malloc(num_hands * sizeof(stdint.uint32_t))
        stdint.uint32_t *new_sum_hands_e = <stdint.uint32_t *>PyMem_Malloc(num_hands * sizeof(stdint.uint32_t))

    if len_board < 5:
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

    PyMem_Free(new_sum_hands_a)
    PyMem_Free(new_sum_hands_b)
    PyMem_Free(new_sum_hands_c)
    PyMem_Free(new_sum_hands_d)
    PyMem_Free(new_sum_hands_e)
    return


cpdef double[:] evaluate_hands_c(int[:] hands, int[:] board=array('i', [])):
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
        double *results = <double *>PyMem_Malloc(len_hands * sizeof(double))  # array to fill with win,tie hands
        int i
    # hands and board to one array dead cards
    for i in range(len_hands):
        results[i] = 0.0  # start results to 0
        dead_cards[i] = hands[i]
    for i in range(len_board):
        dead_cards[len_hands + i] = board[i]
    cdef int[:] dead_cards_py = <int[:len_total]> dead_cards
    cdef int len_deck = create_deck(dead_cards_py, len_total, deck)  # create deck less dead cards
    PyMem_Free(dead_cards)
    # eval board
    sum_board = 53
    for i in range(len_board):
        tmp_sum = sum_board + board[i]
        sum_board = handdat[tmp_sum]
    # eval hands
    for i in range(num_hands):
        # hand card 0
        tmp_sum = sum_board + hands[i * 2]
        sum_hands[i] = handdat[tmp_sum]
        # hand card 1
        tmp_sum = sum_hands[i] + hands[i * 2 + 1]
        sum_hands[i] = handdat[tmp_sum]
    # brute force fill board with all cards and eval it
    if len_board < 5:
        _create_boards(deck, len_deck, sum_hands, num_hands, len_board, results)
    else:  # eval board, each hand in results
        eval_hands(sum_hands, num_hands, results)

    PyMem_Free(sum_hands)
    PyMem_Free(deck)

    cdef double[:] results_py = <double[:len_hands]> results  # C array to memory view
    return results_py