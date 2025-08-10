import numpy as np
from poker_eval_faster import distribution_one_hand_vs_all


def test_distributions_turn_and_flop_shapes():
    d_turn = distribution_one_hand_vs_all(['2s', '6s'], ['Qs', 'Ks', '7c', 'Ah'])
    assert isinstance(d_turn, np.ndarray)
    assert d_turn.shape == (53,)

    d_flop = distribution_one_hand_vs_all(['2s', '6s'], ['Qs', 'Ks', '7c'])
    assert isinstance(d_flop, np.ndarray)
    assert d_flop.shape == (53, 53)


