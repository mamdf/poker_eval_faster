from twoplustwo_eval import evaluate_hands, evaluate_one_hand_vs_all
import numpy as np
import pytest

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

ps_eval_hands = [
    [[['9c', '8c'], ['Tc', 'Td']], ['Qh', 'Jh', '8s'], [15.45, 84.55]],
    [[['Kc', 'Qc'], ['6c', '6d'], ['9s', '8s']], [], [41.25, 28.18, 30.57]],
    [[['Kc', 'Qc'], ['6c', '6d'], ['Kd', 'Kh']], [], [12.85, 19.87, 67.27]]
    ]


@pytest.mark.parametrize('hand, board, expected_pct', ps_hand_vs_all)
def test_evaluate_one_hand_vs_all(hand, board, expected_pct):
    result = evaluate_one_hand_vs_all(hand, board)[0]
    assert round(result * 100, 2) == expected_pct


def test_distributions():
    result, distributions = evaluate_one_hand_vs_all(['2s', '6s'], ['Qs', 'Ks', '7c', 'Ah'])
    assert round(result * 100, 1) == 29.8
    expected = [0,  # rivers
                43.94, 43.94, 43.94, 0,  # 2 (cdhs)
                5.71,  5.71,  5.71,  97.27,  # 3
                5.71,  5.71,  5.71,  97.27,  # 4
                5.71,  5.71,  5.71,  97.27,  # 5
                46.06,  46.06,  46.06,  0,  # 6
                0, 25.10, 25.10, 94.65,  # 7
                6.52, 6.52, 6.52, 97.47,  # 8
                6.52, 6.52, 6.52, 97.47,  # 9
                6.52, 6.52, 6.52, 97.47,  # T
                6.52, 6.52, 6.52, 97.47,  # J
                6.52, 6.52, 6.52, 0,  # Q
                6.52, 6.52, 6.52, 0,  # K
                6.52, 6.52, 0, 94.65]  # A
    for i in range(len(distributions)):
        card_res = distributions[i]
        assert round(card_res * 100, 2) == expected[i]

    result, distributions = evaluate_one_hand_vs_all(['2s', '6s'], ['Qs', 'Ks', '7c'])
    assert round(result * 100, 1) == 46.0
    expected = [0,   # mean river
                61.05, 61.79, 61.79, 0,  # 2 (cdhs)
                28.25, 28.45, 28.45, 93.15,  # 3
                28.11, 28.31, 28.31, 93.15,  # 4
                28.01, 28.20, 28.20, 93.15,   # 5
                63.01, 63.72, 63.72, 0,  # 6
                0, 35.67, 35.67, 88.60,  # 7
                28.39, 28.60, 28.60, 93.93,  # 8
                28.58, 28.79, 28.79, 93.93,  # 9
                28.76, 28.97, 28.97, 93.93,  # T
                29.11, 29.32, 29.32, 93.93,  # J
                31.03, 31.29, 31.29, 0,  # Q
                31.03, 31.29, 31.29, 0,  # K
                29.63, 29.85, 29.85, 93.93  # A
                ]
    for i, river_mean in enumerate(distributions[0]):
        assert round(river_mean * 100, 2) == expected[i]


@pytest.mark.parametrize('hands, board, expected_pct', ps_eval_hands)
def test_evaluate_hands(hands, board, expected_pct):
    result = evaluate_hands(hands, board)
    assert [round(i * 100, 2) for i in result] == expected_pct
