# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, infer_types=True
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport cython
from libc cimport stdint
import numpy as np

from poker_eval_faster.eval_cython.main cimport create_deck
from poker_eval_faster.eval_cython.main cimport handdat
from poker_eval_faster.eval_cython.common cimport fold_cards

cdef int EVAL_START = 53
cdef int WIN = 0
cdef int TIE = 1
cdef int LOSS = 2


cdef void _two_random_on_board(int hero0, int hero1, int* deck, int n,
                               stdint.uint32_t board_rank,
                               stdint.uint64_t* counts) noexcept nogil:
    # Hands are edges between cards. Ordered pairs of disjoint edges in a
    # subset S: |S|*(|S|-1) - sum_card degree*(degree-1).
    cdef int lower_degree[53]
    cdef int eligible_degree[53]
    cdef int i, j, a, b
    cdef stdint.uint32_t hero_rank = handdat[handdat[board_rank + hero0] + hero1]
    cdef stdint.uint32_t prefix, rank
    cdef stdint.uint64_t lower = 0, eligible = 0, wins, unbeaten, total
    for i in range(53):
        lower_degree[i] = 0
        eligible_degree[i] = 0
    for i in range(n):
        a = deck[i]
        prefix = handdat[board_rank + a]
        for j in range(i + 1, n):
            b = deck[j]
            rank = handdat[prefix + b]
            if rank <= hero_rank:
                eligible += 1
                eligible_degree[a] += 1
                eligible_degree[b] += 1
                if rank < hero_rank:
                    lower += 1
                    lower_degree[a] += 1
                    lower_degree[b] += 1
    wins = lower * (lower - 1) if lower else 0
    unbeaten = eligible * (eligible - 1) if eligible else 0
    for i in range(53):
        wins -= lower_degree[i] * (lower_degree[i] - 1)
        unbeaten -= eligible_degree[i] * (eligible_degree[i] - 1)
    total = (<stdint.uint64_t>n * (n - 1) * (n - 2) * (n - 3)) // 4
    counts[0] += wins
    counts[1] += unbeaten - wins
    counts[2] += total - unbeaten


cpdef object evaluate_one_hand_vs_two_random_c(int[:] hero, int[:] board, int[:] dead):
    """Validated inputs only; exact win/tie/loss EVENTS, not pot shares."""
    cdef bint blocked[53]
    cdef int deck[52]
    cdef int rivals[52]
    cdef int i, a, b, n = 0, m, missing = 5 - board.shape[0]
    cdef stdint.uint32_t prefix = 53, turn, river
    cdef stdint.uint64_t counts[3]
    for i in range(53):
        blocked[i] = False
    blocked[hero[0]] = True
    blocked[hero[1]] = True
    for i in range(board.shape[0]):
        blocked[board[i]] = True
        prefix = handdat[prefix + board[i]]
    for i in range(dead.shape[0]):
        blocked[dead[i]] = True
    for i in range(1, 53):
        if not blocked[i]:
            deck[n] = i
            n += 1
    counts[0] = counts[1] = counts[2] = 0
    with nogil:
        if missing == 0:
            _two_random_on_board(hero[0], hero[1], deck, n, prefix, counts)
        else:
            for a in range(n):
                turn = handdat[prefix + deck[a]]
                if missing == 1:
                    m = 0
                    for i in range(n):
                        if i != a:
                            rivals[m] = deck[i]
                            m += 1
                    _two_random_on_board(hero[0], hero[1], rivals, m, turn, counts)
                else:
                    for b in range(a + 1, n):
                        river = handdat[turn + deck[b]]
                        m = 0
                        for i in range(n):
                            if i != a and i != b:
                                rivals[m] = deck[i]
                                m += 1
                        _two_random_on_board(hero[0], hero[1], rivals, m, river, counts)
    return (int(counts[0]), int(counts[1]), int(counts[2]))


@cython.cdivision(True)
cdef inline double _hand_equity(double wins, double ties, double losses) nogil:
    cdef double total = wins + 2.0 * ties + losses
    if total == 0.0:
        return 0.0
    return (wins + ties) / total


@cython.cdivision(True)
cpdef double hand_to_equity(double[:] results):
    return _hand_equity(results[WIN], results[TIE], results[LOSS])

@cython.cdivision(True)
cdef inline double _hand_to_equity_c(double results[]) nogil:
    return _hand_equity(results[WIN], results[TIE], results[LOSS])


cdef void _all_hands_create_boards(int[:] cards, int len_cards, stdint.uint32_t eval_hand, stdint.uint32_t eval_board,
                             double results[], double[:,:] distributions):
    cdef:
        stdint.uint32_t eval_board_turn, eval_board_river
        stdint.uint32_t eval_hand_turn, eval_hand_river
        int [7] dead_cards = [0,0,0,0,0,0,0]
        int *deck = <int *> PyMem_Malloc((52 - len_cards) * sizeof(int))
        int len_deck = create_deck(cards, len_cards, deck)  # call func to create deck (rival_cards)
        int a, b, i
        double n_simulations = 46.0 if len_cards == 6 else 47.0  # number simulations (turn or river)
        double tmp_results_turn[3]
        double tmp_results_river[3]
    for i in range(len_cards):
        dead_cards[i] = cards[i]

    cdef double eq
    for a in range(len_deck):
        eval_board_turn = handdat[eval_board + deck[a]]
        eval_hand_turn = handdat[eval_hand + deck[a]]
        if len_cards == 6:  # just complete river card
            dead_cards[6] = deck[a]  # add river card
            for i in range(3):
                tmp_results_turn[i] = 0.0
            _evaluate_all_rival_hands(dead_cards, 7, eval_hand_turn, eval_board_turn, tmp_results_turn)
            eq = _hand_to_equity_c(tmp_results_turn)
            distributions[0, deck[a]] = eq
            results[0] += tmp_results_turn[0]
            results[1] += tmp_results_turn[1]
            results[2] += tmp_results_turn[2]
            continue

        dead_cards[5] = deck[a]  # add turn card
        for b in range(a+1, len_deck):  # complete river after simulate turn
            eval_board_river = handdat[eval_board_turn + deck[b]]
            eval_hand_river = handdat[eval_hand_turn + deck[b]]
            dead_cards[6] = deck[b]  # add river card
            for i in range(3):
                tmp_results_river[i] = 0.0
            _evaluate_all_rival_hands(dead_cards, 7, eval_hand_river, eval_board_river,
                                                    tmp_results_river)
            eq = _hand_to_equity_c(tmp_results_river)
            distributions[deck[a], deck[b]] = eq
            distributions[deck[b], deck[a]] = eq
            results[0] += tmp_results_river[0]
            results[1] += tmp_results_river[1]
            results[2] += tmp_results_river[2]

    PyMem_Free(deck)
    return


cdef void _evaluate_all_rival_hands(int[:] dead_cards, int len_dead_cards, stdint.uint32_t eval_hand,
                                   stdint.uint32_t eval_board, double results[]):
    """
    Evaluate all rival hands vs eval_hand in one board
    :param results: array to save number of win and tie per hand
    :return None
    """
    cdef:
        stdint.uint32_t tmp_sum, first_eval_rival, eval_rival
        double win=0.0, loss=0.0, tie=0.0
        int c1, c2, card1, card2
        int *rival_cards = <int *> PyMem_Malloc((52 - len_dead_cards) * sizeof(int))
        int num_cards = create_deck(dead_cards, len_dead_cards, rival_cards)  # call func to create deck (rival_cards)
    # evaluate all rival hands possibles
    with nogil:
        for c1 in range(num_cards):
            card1 = rival_cards[c1]
            tmp_sum = eval_board + card1
            first_eval_rival = handdat[tmp_sum]
            for c2 in range(c1 + 1, num_cards):
                card2 = rival_cards[c2]
                tmp_sum = first_eval_rival + card2
                eval_rival = handdat[tmp_sum]
                if len_dead_cards < 7:
                    eval_rival = handdat[eval_rival]
                if eval_hand > eval_rival:
                    win += 1.0
                elif eval_hand == eval_rival:
                    tie += 1.0
                else:
                    loss += 1.0

    PyMem_Free(rival_cards)
    results[WIN] += win
    results[TIE] += tie / 2
    results[LOSS] += loss


cpdef object evaluate_one_hand_vs_all_c(int[:] cards, double[:,:] distributions, incomplete_board=False):
    """
    Evaluate one hand vs all other hands in one specific board
    :param cards: Array int where hand[:2] and board[2:]
    :param distributions: np.zeros([53,53]) distribution equity per round per card
    :param incomplete_board: if False and board < 5 cards, complete it with all possible combinations
    :return: probabilities to [win, tie], distributions
    """
    cdef:
        stdint.uint32_t tmp_sum, eval_board, eval_hand
        int i
        int len_cards = cards.shape[0]
        object results_np = np.zeros(3, dtype=np.float64)
        cdef double[::1] results = results_np
    # eval board
    eval_board = fold_cards(EVAL_START, cards, 2, len_cards)
    for i in range(3):
        results[i] = 0.0
    # eval hand (sum hand over board)
    eval_hand = fold_cards(eval_board, cards, 0, 2)

    if len_cards < 7 and incomplete_board:  # evaluate hands strength on current board
        eval_hand = handdat[eval_hand]
        _evaluate_all_rival_hands(cards, len_cards, eval_hand, eval_board, &results[0])
    elif len_cards == 7:
        _evaluate_all_rival_hands(cards, len_cards, eval_hand, eval_board, &results[0])
    else:
        _all_hands_create_boards(cards, len_cards, eval_hand, eval_board, &results[0], distributions)

    return results_np
