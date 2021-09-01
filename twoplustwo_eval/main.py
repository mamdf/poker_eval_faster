from twoplustwo_eval import evaluate_one_hand_vs_all_c, hand_to_equity
from twoplustwo_eval import evaluate_hands_c, hands_to_equity
from twoplustwo_eval import evaluate_c
from typing import List, Tuple
import numpy as np

DECK = [r + s for r in '23456789TJQKA' for s in 'cdhs']
CARDS_TO_INT = {card: i for i, card in enumerate(DECK, start=1)}
RANKING = [None, "NOPAIR", "PAIR", "DOUBLES", "TRIPS", "STRAIGHT", "FLUSH", "FULL", "QUADS", "STRAIGHT_FLUSH"]


def cards_to_int_array(cards: List[str]):
    return np.array([CARDS_TO_INT[c] for c in cards], dtype='int32')


def cards_to_array(cards: List[int]):
    return np.array(cards, dtype='int32')


def int_to_cards(cards: List[int]):
    return [DECK[i-1] for i in cards]


def ranking_to_category(rank: int) -> Tuple[int, str]:
    rank = rank >> 12  # rank to num category
    return rank, RANKING[rank]


def evaluate_rank(board: List, hand: List = []):
    if type(board[0]) is str:
        cards = cards_to_int_array(hand + board)
    else:
        cards = cards_to_array(hand + board)
    rank = evaluate_c(cards)
    return rank


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
    distributions = np.zeros([53, 53])  # dim 0 is used in flop to fill turn means
    distributions_flop = []
    # create mask for no cards in sort equity
    mask = np.ones(53, dtype='bool')
    mask[cards] = False
    mask[0] = False  # (cards 1-53)
    # evaluate
    ev = evaluate_one_hand_vs_all_c(cards, distributions, incomplete_board=False)

    if len(board) == 3:
        for i in range(1, len(distributions)):  # equity in each turn card (mean rivers)
            if i in cards:  # dist_turn = zeros
                continue
            dist_turn = distributions[i]
            mask_turn = mask.copy()
            mask_turn[i] = False
            valid_cards = dist_turn[mask_turn]
            distributions[0][i] = valid_cards.mean()
            if sort_distributions:
                distributions_flop.append(np.sort(valid_cards))

        if sort_distributions:
            arg_mean = np.argsort(distributions[0][mask])
            return np.array(distributions_flop)[arg_mean]
        else:
            return distributions
    elif len(board) == 4:
        if sort_distributions:
            return np.sort(distributions[0][mask])
        else:
            return distributions[0]
    else:
        return ev


if __name__ == '__main__':
    distribution_one_hand_vs_all(['2s', '6s'], ['Qs', 'Ks', '7c'], sort_distributions=True)