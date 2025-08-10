import pytest
import numpy as np

from poker_eval_faster import evaluate_hands_c, hands_to_equity, cards_to_int_array


def test_evaluate_hands_c_basic():
    hands = cards_to_int_array(['9c', '8c', 'Tc', 'Td'])
    board = cards_to_int_array(['Qh', 'Jh', '8s'])
    res = evaluate_hands_c(hands, board)
    eq = hands_to_equity(np.array(res))
    assert isinstance(eq, list)
    assert len(eq) == 2


