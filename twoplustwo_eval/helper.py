from typing import List
from array import array

DECK = [r + s for r in '23456789TJQKA' for s in 'cdhs']
CARDS_TO_INT = {card: i for i, card in enumerate(DECK, start=1)}


def cards_to_int(cards: List[str]):
    return array('i', [CARDS_TO_INT[c] for c in cards])


def int_to_cards(cards: List[int]):
    return [DECK[i-1] for i in cards]


def hands_to_int(hands):
    return [[CARDS_TO_INT[c] - 1 for c in hand] for hand in hands]
