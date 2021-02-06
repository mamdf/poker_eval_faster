from twoplustwo_eval.evaluate import evaluate_one_hand_vs_all_c, evaluate_hands_c, hand_to_equity, hands_to_equity
from twoplustwo_eval.helper import cards_to_int, int_to_cards


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

