from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport cython
from libc cimport stdint

from twoplustwo_eval.eval_cython.main cimport handdat
from twoplustwo_eval.eval_cython.main cimport create_deck


@cython.cdivision(True)
cpdef double hand_to_equity(double[:] results):
    cdef:
        double total = results[0] + results[1] * 2 + results[2]
        double won = results[0] + results[1]
    return won / total


cdef void _all_hands_create_boards(int[:] cards, int len_cards, stdint.uint32_t eval_hand, stdint.uint32_t eval_board,
                             double[:] results):
    cdef:
        stdint.uint32_t eval_board_turn, eval_board_river
        stdint.uint32_t eval_hand_turn, eval_hand_river
        int [7] dead_cards = [0,0,0,0,0,0,0]
        int *deck = <int *> PyMem_Malloc((52 - len_cards) * sizeof(int))
        int len_deck = create_deck(cards, len_cards, deck)  # call func to create deck (rival_cards)
        int a, b, i
        double n_simulations = 46.0 if len_cards == 6 else 47.0  # number simulations (turn or river)
    for i in range(len_cards):
        dead_cards[i] = cards[i]

    for a in range(len_deck):
        eval_board_turn = handdat[eval_board + deck[a]]
        eval_hand_turn = handdat[eval_hand + deck[a]]
        if len_cards == 6:
            dead_cards[6] = deck[a]
            _evaluate_all_rival_hands(dead_cards, 7, eval_hand_turn, eval_board_turn, results)
        else:
            dead_cards[5] = deck[a]
            for b in range(a+1, len_deck):
                eval_board_river = handdat[eval_board_turn + deck[b]]
                eval_hand_river = handdat[eval_hand_turn + deck[b]]
                dead_cards[6] = deck[b]
                _evaluate_all_rival_hands(dead_cards, 7, eval_hand_river, eval_board_river, results)

    PyMem_Free(deck)
    return


cdef void _evaluate_all_rival_hands(int[:] dead_cards, int len_dead_cards, stdint.uint32_t eval_hand,
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


cpdef double[:] evaluate_one_hand_vs_all_c(int[:] cards):
    """
    Evaluate one hand vs all other hands in one specific board
    :param cards: Array int where hand[:2] and board[2:]
    :return: probabilities to [win, tie]
    """
    cdef:
        stdint.uint32_t tmp_sum, eval_board, eval_hand
        int i
        int len_cards = cards.shape[0]
        double[3] results = [0.0, 0.0, 0.0]
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
        _evaluate_all_rival_hands(cards, len_cards, eval_hand, eval_board, results)
    else:
        _all_hands_create_boards(cards, len_cards, eval_hand, eval_board, results)

    return results



