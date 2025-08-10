import numpy as np
import pytest

from poker_eval_faster import evaluate_one_hand_vs_all_c, cards_to_int_array


@pytest.mark.parametrize('hand, board', [
    (['5c', '2d'], ['2c', '2h', '5h', '7h', '7s']),
    (['Tc', 'Qc'], ['Ad', 'Jh', '3s']),
])
def test_evaluate_one_hand_vs_all_c_import_and_call(hand, board):
    cards = cards_to_int_array(hand + board)
    d = np.zeros((53, 53), dtype=float)
    res = evaluate_one_hand_vs_all_c(cards, d, False)
    assert isinstance(res, np.ndarray) or hasattr(res, '__array__')


