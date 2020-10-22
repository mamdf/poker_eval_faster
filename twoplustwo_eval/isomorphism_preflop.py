import numpy as np
from typing import List


def _suit_config(cards_int, last_suit=0, config=None):
    if not len(cards_int):  # base
        return []

    suit_idx = cards_int[0] % 4
    rank = cards_int[0] - suit_idx  # rank club
    if config and config.get(suit_idx) is not None:  # suit's already in config
        new_card = rank + config[suit_idx]
    else:
        if config:
            last_suit += 1
        else:
            config = {}
        assert last_suit < 4, f'{rank}, {config}'
        new_card = rank + last_suit
        config[suit_idx] = last_suit

    return [new_card] + _suit_config(cards_int[1:], last_suit, config)


def fix_hole_cards(cards_int: List[List[int]]):
    cards_int = np.sort(cards_int, axis=1)  # hole card sorted c1 < c2
    cards_int = np.sort(cards_int, axis=0)  # p1 < p2 < p3

    result = _suit_config(cards_int.reshape(cards_int.size, ))
    result = np.sort(np.array_split(result, cards_int.shape[0]), axis=1)
    result = result.reshape((result.size,))
    return np.asarray([i+1 for i in result], dtype='int')
