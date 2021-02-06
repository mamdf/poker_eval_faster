from twoplustwo_eval import evaluate_one_hand_vs_all_c, hand_to_equity
from twoplustwo_eval import evaluate_hands_c, hands_to_equity
from typing import List, Tuple
from array import array

DECK = [r + s for r in '23456789TJQKA' for s in 'cdhs']
CARDS_TO_INT = {card: i for i, card in enumerate(DECK, start=1)}


def cards_to_int(cards: List[str]):
    return array('i', [CARDS_TO_INT[c] for c in cards])


def int_to_cards(cards: List[int]):
    return [DECK[i-1] for i in cards]


def evaluate_hands(hands, board=None):
    """
    :param hands: List[List[str, str]]
    :param board: List[str]
    :return:
    """
    hands_cards = [card for hand in hands for card in hand]
    hands_cards = cards_to_int(hands_cards)
    if board:
        board_cards = cards_to_int(board)
        ev = evaluate_hands_c(hands_cards, board_cards)
    else:
        ev = evaluate_hands_c(hands_cards)

    return hands_to_equity(ev)


def evaluate_one_hand_vs_all(hand, board):
    """
    :param hand:
    :param board:
    :return:
    """
    cards = cards_to_int(hand + board)
    ev = evaluate_one_hand_vs_all_c(cards)
    return hand_to_equity(ev)

