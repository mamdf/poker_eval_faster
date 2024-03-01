from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport cython
from libc cimport stdint

from poker_eval_faster.eval_cython.main cimport handdat
from poker_eval_faster.eval_cython.main cimport create_deck


@cython.cdivision(True)
cpdef double hand_to_equity(double[:] results):
    cdef:
        double total = results[0] + results[1] * 2 + results[2]
        double won = results[0] + results[1]
    return won / total

@cython.cdivision(True)
cdef double _hand_to_equity_c(double results[]):
    cdef:
        double total = results[0] + results[1] * 2 + results[2]
        double won = results[0] + results[1]
    return won / total


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
        double * tmp_results_turn = <double *> PyMem_Malloc(3 * sizeof(double))
        double * tmp_results_river = <double *> PyMem_Malloc(3 * sizeof(double))
    for i in range(len_cards):
        dead_cards[i] = cards[i]

    for a in range(len_deck):
        eval_board_turn = handdat[eval_board + deck[a]]
        eval_hand_turn = handdat[eval_hand + deck[a]]
        if len_cards == 6:  # just complete river card
            dead_cards[6] = deck[a]  # add river card
            for i in range(3):  # reset to 0
                tmp_results_turn[i] = 0.0
            _evaluate_all_rival_hands(dead_cards, 7, eval_hand_turn, eval_board_turn, tmp_results_turn)
            distributions[0, deck[a]] = _hand_to_equity_c(tmp_results_turn)
            results[0] += tmp_results_turn[0]
            results[1] += tmp_results_turn[1]
            results[2] += tmp_results_turn[2]
            continue

        dead_cards[5] = deck[a]  # add turn card
        for b in range(a+1, len_deck):  # complete river after simulate turn
            eval_board_river = handdat[eval_board_turn + deck[b]]
            eval_hand_river = handdat[eval_hand_turn + deck[b]]
            dead_cards[6] = deck[b]  # add river card
            for i in range(3):  # reset to 0
                tmp_results_river[i] = 0.0
            _evaluate_all_rival_hands(dead_cards, 7, eval_hand_river, eval_board_river,
                                                    tmp_results_river)
            distributions[deck[a], deck[b]] = _hand_to_equity_c(tmp_results_river)
            distributions[deck[b], deck[a]] = distributions[deck[a], deck[b]]  # repeated is not simulate (4h5h == 5h4h)
            results[0] += tmp_results_river[0]
            results[1] += tmp_results_river[1]
            results[2] += tmp_results_river[2]

    PyMem_Free(deck)
    PyMem_Free(tmp_results_turn)
    PyMem_Free(tmp_results_river)
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
    for c1 in range(num_cards):
        card1 = rival_cards[c1]
        tmp_sum = eval_board + card1
        first_eval_rival = handdat[tmp_sum]  # sum only first card
        for c2 in range(c1 + 1, num_cards):
            card2 = rival_cards[c2]
            tmp_sum = first_eval_rival + card2
            eval_rival = handdat[tmp_sum]
            if len_dead_cards < 7:  # or len board si le sumo rangos a dead cards
                eval_rival = handdat[eval_rival]
            # evaluate counter
            if eval_hand > eval_rival:
                win += 1.0
            elif eval_hand == eval_rival:
                tie += 1.0
            else:
                loss += 1.0

    PyMem_Free(rival_cards)
    results[0] += win
    results[1] += tie / 2
    results[2] += loss


cpdef double[:] evaluate_one_hand_vs_all_c(int[:] cards, double[:,:] distributions, incomplete_board=False):
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
        double *results = <double *>PyMem_Malloc(3 * sizeof(double))
    # eval board
    eval_board = 53
    for i in range(3):
        results[i] = 0.0  # start results to 0
    for i in range(2, len_cards):  # board is the same for all hands, store sum
        tmp_sum = eval_board + cards[i]
        eval_board = handdat[tmp_sum]
    # eval hand
    eval_hand = eval_board  # starting in board sum
    for i in range(2):
        tmp_sum = eval_hand + cards[i]
        eval_hand = handdat[tmp_sum]

    if len_cards < 7 and incomplete_board:  # evaluate hands strength on current board
        eval_hand = handdat[eval_hand]
        _evaluate_all_rival_hands(cards, len_cards, eval_hand, eval_board, results)
    elif len_cards == 7:
        _evaluate_all_rival_hands(cards, len_cards, eval_hand, eval_board, results)
    else:
        _all_hands_create_boards(cards, len_cards, eval_hand, eval_board, results, distributions)

    cdef double[:] results_py = <double[:3]> results  # C array to memory view
    return results_py
