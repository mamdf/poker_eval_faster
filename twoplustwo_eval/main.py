from twoplustwo_eval import evaluate_one_hand_vs_all_c, hand_to_equity
from twoplustwo_eval import evaluate_hands_c, hands_to_equity
from typing import List, Tuple
import numpy as np

DECK = [r + s for r in '23456789TJQKA' for s in 'cdhs']
CARDS_TO_INT = {card: i for i, card in enumerate(DECK, start=1)}


def cards_to_int_array(cards: List[str]):
    return np.array([CARDS_TO_INT[c] for c in cards], dtype='int32')


def cards_to_array(cards: List[int]):
    return np.array(cards, dtype='int32')


def int_to_cards(cards: List[int]):
    return [DECK[i-1] for i in cards]


def evaluate_hands(hands, board=None, eq=True, incomplete_board=False) -> List[float]:
    """
    :param hands: List[List[str, str]] or List[List[int, int]]
    :param board: List[str] or List[int] or None
    :param eq: return equity or combos (win, win... tie, tie...)
    :return: List[float] equity hands or combos
    """
    hands_cards = [card for hand in hands for card in hand]
    if type(hands_cards[0]) is str:  # ex. Ac, Kc
        hands_cards = cards_to_int_array(hands_cards)
    else:  # ex. 45, 50
        hands_cards = cards_to_array(hands_cards)
    len_cards = hands_cards.size

    if board:
        if type(board[0]) is str:  # ex. Ac
            board_cards = cards_to_int_array(board)
        else:  # ex. 45
            board_cards = cards_to_array(board)
        len_cards += board_cards.size
        if np.unique(hands_cards).size + np.unique(board_cards).size != len_cards:  # repeated cards
            ev = np.zeros(len_cards)
        else:
            ev = evaluate_hands_c(hands_cards, board_cards)
    else:
        if np.unique(hands_cards).size != len_cards:   # repeated cards
            ev = np.zeros(len_cards)
        else:
            ev = evaluate_hands_c(hands_cards)

    if eq:
        return hands_to_equity(ev)
    else:
        return list(ev)


def evaluate_one_hand_vs_all(hand, board, eq=True, incomplete_board=False):
    """
    :param hand: List[str, str] or List[int, int]
    :param board: List[str] or List[int]
    :param eq: return equity or combos (win, tie, lose)
    :param incomplete_board: if False and board < 5 cards, complete it with all possible combinations
    :return: List[float] equity hand or combos
    """
    if type(hand[0]) is str:
        cards = cards_to_int_array(hand + board)
    else:
        cards = cards_to_array(hand + board)
    distributions = np.zeros([53, 53])
    ev = evaluate_one_hand_vs_all_c(cards, distributions, incomplete_board)
    if eq:
        result = hand_to_equity(ev)
    else:
        result = list(ev)

    return result


def distribution_one_hand_vs_all(hand, board, sort_distributions=False):
    if type(hand[0]) is str:
        cards = cards_to_int_array(hand + board)
    else:
        cards = cards_to_array(hand + board)
    distributions = np.zeros([53, 53])
    distributions_flop = []
    ev = evaluate_one_hand_vs_all_c(cards, distributions, incomplete_board=False)

    if len(board) == 3:
        for i in range(len(distributions)):  # equity in each turn card (mean rivers)
            dist_turn = distributions[i]
            valid_cards = dist_turn[dist_turn.nonzero()]
            if valid_cards.size:
                distributions[0][i] = valid_cards.mean()
                if sort_distributions:
                    distributions_flop.append(np.sort(valid_cards))

        if sort_distributions:
            arg_mean = np.argsort(distributions[0][distributions[0].nonzero()])
            return np.array(distributions_flop)[arg_mean]
        else:
            return distributions
    elif len(board) == 4:
        if sort_distributions:
            return np.sort(distributions[0][distributions[0].nonzero()])
        else:
            return distributions[0]
    else:
        return ev
