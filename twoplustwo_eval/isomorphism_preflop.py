from typing import List, Tuple


def sort_axis1(hands):
    sort_cards = []
    for h in hands:
        if h[0] > h[1]:
            sort_cards.append((h[1], h[0]))
        else:
            sort_cards.append((h[0], h[1]))

    return sort_cards


def _suit_config(hands_int, last_suit=0, config=None):
    if not len(hands_int):  # base
        return []

    new_hand = []
    for card in hands_int[0]:
        suit_idx = (card - 1) % 4
        rank = card - suit_idx  # rank club
        if config and config.get(suit_idx) is not None:  # suit's already in config
            new_hand.append(rank + config[suit_idx])
        else:
            if config:
                last_suit += 1
            else:
                config = {}
            assert last_suit < 4, f'{rank}, {config}'
            new_hand.append(rank + last_suit)
            config[suit_idx] = last_suit

    return [new_hand] + _suit_config(hands_int[1:], last_suit, config)


def fix_hole_cards(hands_int: List[List[int]]):
    """
    Canonical pre-flop hands, ex: AcKc QdJd == Ahch QsJs
    :param hands_int: List[List[int]]
    :return: List[int] canonical cards, List[int] arg index sorted
    """
    arg_index = []
    hands_int = sort_axis1(hands_int)  # hole card sorted c1 < c2
    sort_hands = sorted(hands_int)  # p1 < p2 < p3
    for hand in hands_int:
        arg_index.append(sort_hands.index(hand))

    result = _suit_config(sort_hands)
    result = sort_axis1(result)  # it has not sorted yet, sorted (ex: AdAc -> AcAd)
    result = [i for hand in result for i in hand]  # flatten
    return result, arg_index
