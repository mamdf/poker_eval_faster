import pytest
from typing import cast

from poker_eval_faster import evaluate_one_hand_vs_all


ps_hand_vs_all = [
    [['5c', '2d'], ['2c', '2h', '5h', '7h', '7s'], 88.99],
    [['Qc', '2d'], ['5c', 'Tc', '3d', '4d', '9h'],  12.17],
    [['Ac', '4d'], ['8c', '4h', '5h', 'Jh', '2s'],  50.61],
    [['4c', '5d'], ['2c', '7c', 'Qc', 'Qh', '6s'],  2.22],
    [['6c', '8d'], ['9c', '2h', 'Th', '2s', '9s'],  10.15],
    [['6c', '6d'], ['9c', '7d', 'Jh', 'Ah', '6s'],  95.56],
    [['9c', 'Jc'], ['5d', '9d', 'Kd', '4h'],  65.48],
    [['3c', '4c'], ['6c', '6d', 'Kd', '7h'],  19.37],
    [['Jc', '3d'], ['Ac', '9d', '7h', 'Qh'],  31.12],
    [['6c', 'Kd'], ['3c', 'Qc', '5h', '7h'],  37.3],
    [['Tc', '9d'], ['2c', 'Ac', 'Qh', '7s'],  28.87],
    [['5c', '8d'], ['8c', 'Tc', '3h', 'Jh'],  57.26],
    [['Tc', 'Qc'], ['Ad', 'Jh', '3s'], 52.4],
    [['8c', '6d'], ['9c', '5d', '3h'], 36.16],
    [['5c', '9d'], ['9c', 'Qc', 'Jh'], 58.99],
    [['7c', 'Kc'], ['7d', 'Jd', '6h'], 67.11],
    [['5c', '7d'], ['Ac', '5h', '7h'], 82.59],
    [['Kc', 'Jd'], ['2c', '4c', '2d'], 52.19],
]


@pytest.mark.parametrize('hand, board, expected_pct', ps_hand_vs_all)
def test_evaluate_one_hand_vs_all(hand, board, expected_pct):
    result = evaluate_one_hand_vs_all(hand, board)
    assert round(cast(float, result) * 100, 2) == expected_pct


