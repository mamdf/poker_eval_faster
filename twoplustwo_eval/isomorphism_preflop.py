import numpy as np
from typing import List


def _suit_config(hands_int, last_suit=0, config=None):
    if not len(hands_int):  # base
        return []

    new_hand = []
    for card in hands_int[0]:
        suit_idx = card % 4
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
    hands_int = np.sort(hands_int, axis=1)  # hole card sorted c1 < c2
    hands_int = np.sort(hands_int, axis=0)  # p1 < p2 < p3

    result = _suit_config(hands_int)
    result = np.sort(result, axis=1)  # it has not sorted yet, sorted (ex: AdAc -> AcAd)
    return np.asarray([i+1 for hand in result for i in hand], dtype='int32')  # flatten
