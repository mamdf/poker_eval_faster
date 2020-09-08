import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from libc cimport stdint

path = "/home/marcos/Gits/poker_projects/poker_hand_evaluator/data/HandRanks.dat"
dat = np.fromfile(path, dtype=np.uint32)
cdef stdint.uint32_t[:] handdat = dat[:]  # np.array to C


cdef inline sum_new_card(int new_card, stdint.uint32_t sum_hands[], int num_hands, stdint.uint32_t new_sum_hands[]):
    cdef int i
    for i in range(num_hands):
        new_sum_hands[i] = handdat[sum_hands[i] + new_card]

cdef inline eval_hands(stdint.uint32_t sum_hands[], int num_hands, double results[]):
    cdef int i, win_id
    cdef int tie = 0
    cdef stdint.uint32_t max_eval = 0
    for i in range(num_hands):
        if sum_hands[i] > max_eval:
            max_eval = sum_hands[i]
            win_id = i

    for i in range(num_hands):
        if sum_hands[i] == max_eval:
            tie += 1

    if tie > 1:
        for i in range(num_hands):
            if sum_hands[i] == max_eval:
                results[num_hands + i] += 1 / tie
    else:
        results[win_id] += 1



cpdef stdint.uint32_t handStats_C(h):
    """ Takes a hand as an array of strings (as above)
    Returns a dict of the hand's stats.
    value is an integer whose value that can be compared to other hand values.
	Larger number = better hand"""
    cdef stdint.uint32_t p = 53
    cdef stdint.uint32_t suma
    cdef unsigned short i
    cdef unsigned short len_h = len(h)
    for i in range(len_h):
        suma = p + h[i]
        p = handdat[suma]

    if len_h==5 or len_h==6:
        p = handdat[p]

    return p


cdef int create_deck(int[:] dead_cards, int len_dead_cards, int results[]):
    """
    Create all deck cards (int 1 to 53) less dead cards.
    :param dead_cards: memoryview
    :param len_dead_cards: int length dead_cards
    :param results: C array will be append deck cards
    :return: len deck
    """
    cdef:
        int i, j
        int num_cards = 0
        bint flag
    for i in range(1, 53):
        flag = True
        for j in range(len_dead_cards):
            if dead_cards[j] == i:
                flag = False
                break
        if flag:
            results[num_cards] = i
            num_cards += 1

    return num_cards


cdef double _evaluate_all_hands(int[:] dead_cards, int len_dead_cards, stdint.uint32_t eval_hand, stdint.uint32_t eval_board):
    """
    Evaluate all rival hands vs eval_hand in one board
    """
    cdef:
        stdint.uint32_t tmp_sum, first_eval_rival, eval_rival
        int win=0, loss=0, tie=0
        int c1, c2, card1, card2, total
        int *rival_cards = <int *> PyMem_Malloc((52 - len_dead_cards) * sizeof(int))
        int num_cards = create_deck(dead_cards, len_dead_cards, rival_cards)  # call func to create deck (rival_cards)
    # evaluate all rival hands possibles
    for c1 in range(num_cards):
        card1 = rival_cards[c1]
        tmp_sum = eval_board + card1
        first_eval_rival = handdat[tmp_sum]  # sum only first card
        for c2 in range(c1 + 1, num_cards):
            card2 = rival_cards[c2]
            tmp_sum = first_eval_rival + card2
            eval_rival = handdat[tmp_sum]
            if len_dead_cards < 5:
                eval_rival = handdat[eval_rival]
            # evaluate counter
            if eval_hand > eval_rival:
                win += 1
            elif eval_hand == eval_rival:
                tie += 1
            else:
                loss += 1

    PyMem_Free(rival_cards)
    total = win + loss + tie
    return (win + tie / 2) / total


cpdef double evaluate_all_hands(int[:] cards):
    """
    Evaluate one hand vs all other hands in one specific board
    :param cards: Array int where hand[:2] and board[2:]
    :return: probabilities to win
    """
    cdef:
        stdint.uint32_t tmp_sum, eval_board, eval_hand
        int i
        int win=0, loss=0, tie=0
        int len_cards = cards.shape[0]
    # eval board
    eval_board = 53
    for i in range(2, len_cards):  # board is the same for all hands, store sum
        tmp_sum = eval_board + cards[i]
        eval_board = handdat[tmp_sum]
    # eval hand
    eval_hand = eval_board  # starting in board sum
    for i in range(2):
        tmp_sum = eval_hand + cards[i]
        eval_hand = handdat[tmp_sum]
        if len_cards < 5:
            eval_hand = handdat[eval_hand]

    return _evaluate_all_hands(cards, len_cards, eval_hand, eval_board)


cdef void create_boards(int deck[], int len_deck, stdint.uint32_t sum_hands[],
                        int num_hands, int len_board,
                        double results[]):
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


cpdef double[:] evaluate_all_boards(int[:] hands, int[:] board):
    cdef:
        stdint.uint32_t tmp_sum, sum_board
        int len_hands = hands.shape[0]
        int num_hands = len_hands // 2
        int len_board = board.shape[0]
        int len_total = len_board + len_hands
        stdint.uint32_t *sum_hands = <stdint.uint32_t *>PyMem_Malloc(num_hands * sizeof(stdint.uint32_t))
        int *deck = <int *>PyMem_Malloc((52 - len_total) * sizeof(int))
        int *dead_cards = <int *>PyMem_Malloc(len_total * sizeof(int))
        double *results = <double *>PyMem_Malloc(len_hands * sizeof(double))
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
    # brute force fill board with all cards
    if len_board < 5:
        create_boards(deck, len_deck, sum_hands, num_hands, len_board, results)
    else:  # eval board, eval hands in results
        eval_hands(sum_hands, num_hands, results)

    PyMem_Free(sum_hands)
    PyMem_Free(deck)

    cdef double[:] results_py = <double[:len_hands]> results
    return results_py

