import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from pathlib import Path
import cython

from libc cimport stdint

import pickle

path = Path("/home/extra/Data/poker")
path_river_centroid = path / Path('river_centroids.pkl')
path_turn_centroid = path / Path('turn_centroids.pkl')
dat = np.fromfile(path / Path('HandRanks.dat'), dtype=np.uint32)  # eval file two plus two
if path_river_centroid.exists():
    with open(path_river_centroid, 'rb') as f:
        _river_centroids = pickle.load(f)
else:
    _river_centroids = np.zeros((1,1))
if path_turn_centroid.exists():
    with open(path_turn_centroid, 'rb') as f:
        _turn_centroids = pickle.load(f)
else:
    _turn_centroids = np.zeros((1,1))
# np.array to C(memory views)
cdef stdint.uint32_t[:] handdat = dat[:]
cdef double[:] river_centroids = _river_centroids.reshape(_river_centroids.shape[0],)[:]  # river centroids
cdef double[:] turn_centroids = _turn_centroids.reshape(_turn_centroids.shape[0],)[:]  # turn centroids
cdef int len_centroids = _river_centroids.shape[0]


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
cpdef double results_to_ev(double[:] results):
    cdef:
        double total = results[0] + results[1] + results[2]
        double won = results[0] + results[1]
    return won / total


cdef int ehs_distance(double ehs, street='turn'):
    cdef:
        int idx, min_idx
        double min_emd, emd
    for idx in range(len_centroids):
        if street == 'turn':
            emd = abs(ehs - river_centroids[idx])
        else:
            emd = abs(ehs - turn_centroids[idx])
        if idx == 0:
            min_idx = idx
            min_emd = emd
        else:
            if emd < min_emd:
                min_idx = idx
                min_emd = emd

    return min_idx

@cython.cdivision(True)
cdef void ev_clusters(double n_simulations, double[:] results, double clusters[]):
    cdef double ehs
    cdef int idx
    ehs = results_to_ev(results)
    idx = ehs_distance(ehs)
    clusters[idx] += 1 / n_simulations


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

cdef void all_hands_create_boards(int[:] cards, int len_cards, stdint.uint32_t eval_hand, stdint.uint32_t eval_board,
                             double[:] results, double clusters[], bint ehs):
    cdef:
        stdint.uint32_t eval_board_turn, eval_board_river
        stdint.uint32_t eval_hand_turn, eval_hand_river
        int [7] dead_cards = [0,0,0,0,0,0,0]
        int *deck = <int *> PyMem_Malloc((52 - len_cards) * sizeof(int))
        int len_deck = create_deck(cards, len_cards, deck)  # call func to create deck (rival_cards)
        int a, b, i
        double n_simulations = 46.0 if len_cards == 6 else 1081.0  # number simulations (turn or river)
    for i in range(len_cards):
        dead_cards[i] = cards[i]

    for a in range(len_deck):
        eval_board_turn = handdat[eval_board + deck[a]]
        eval_hand_turn = handdat[eval_hand + deck[a]]
        if len_cards == 6:
            dead_cards[6] = deck[a]
            _evaluate_all_hands(dead_cards, 7, eval_hand_turn, eval_board_turn, results)
            if ehs:
                ev_clusters(n_simulations, results, clusters)
        else:
            dead_cards[5] = deck[a]
            for b in range(a+1, len_deck):
                eval_board_river = handdat[eval_board_turn + deck[b]]
                eval_hand_river = handdat[eval_hand_turn + deck[b]]
                dead_cards[6] = deck[b]
                _evaluate_all_hands(dead_cards, 7, eval_hand_river, eval_board_river, results)
                if ehs:
                    ev_clusters(n_simulations, results, clusters)

    PyMem_Free(deck)
    return


cdef void _evaluate_all_hands(int[:] dead_cards, int len_dead_cards, stdint.uint32_t eval_hand,
                                   stdint.uint32_t eval_board, double[:] results):
    """
    Evaluate all rival hands vs eval_hand in one board
    """
    cdef:
        stdint.uint32_t tmp_sum, first_eval_rival, eval_rival
        double win=0.0, loss=0.0, tie=0.0
        int c1, c2, card1, card2
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
            # evaluate counter
            if eval_hand > eval_rival:
                win += 1
            elif eval_hand == eval_rival:
                tie += 1
            else:
                loss += 1

    PyMem_Free(rival_cards)
    results[0] += win
    results[1] += tie / 2
    results[2] += loss
    return


cpdef double[:] evaluate_all_hands(int[:] cards, bint ehs=False):
    """
    Evaluate one hand vs all other hands in one specific board
    :param cards: Array int where hand[:2] and board[2:]
    :param ehs: return ehs clusters
    :return: probabilities to [win, tie]
    """
    cdef:
        stdint.uint32_t tmp_sum, eval_board, eval_hand
        int i
        int len_cards = cards.shape[0]
        double[3] results = [0.0, 0.0, 0.0]
        double *_clusters =  <double *> PyMem_Malloc(len_centroids * sizeof(double))
    # clusters to 0.0
    for i in range(len_centroids):
        _clusters[i] = 0.0
    cdef double[:] clusters = <double[:len_centroids]> _clusters
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

    if len_cards == 7:
        _evaluate_all_hands(cards, len_cards, eval_hand, eval_board, results)
    else:
        all_hands_create_boards(cards, len_cards, eval_hand, eval_board, results, _clusters, ehs)

    if ehs:
        return clusters
    else:
        PyMem_Free(_clusters)
        return results

cdef void create_boards(int deck[], int len_deck, stdint.uint32_t sum_hands[],
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


cpdef double[:] evaluate_all_boards(int[:] hands, int[:] board):
    """
    Evaluate hands vs Board, if board is incomplete (< 5) complete it with all possible cards and eval it.
    :return: [win hand 1, win hand 2, tie hand 1, tie hand 2 ... ]
    """
    cdef:
        stdint.uint32_t tmp_sum, sum_board
        int len_hands = hands.shape[0]
        int num_hands = len_hands // 2
        int len_board = board.shape[0]
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
        create_boards(deck, len_deck, sum_hands, num_hands, len_board, results)
    else:  # eval board, each hand in results
        eval_hands(sum_hands, num_hands, results)

    PyMem_Free(sum_hands)
    PyMem_Free(deck)

    cdef double[:] results_py = <double[:len_hands]> results  # C array to memory view
    return results_py

