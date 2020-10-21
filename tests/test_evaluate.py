from twoplustwo_eval import evaluate_all_hands, cards_to_int, results_to_ev
import pytest

ps_eval = [
    [['5c', '2d', '2c', '2h', '5h', '7h', '7s'], [860, 21, 88], 88.99],
    [['Qc', '2d', '5c', 'Tc', '3d', '4d', '9h'], [116, 4.5, 865], 12.17],
    [['Ac', '4d', '8c', '4h', '5h', 'Jh', '2s'], [498, 3, 486], 50.61],
    [['4c', '5d', '2c', '7c', 'Qc', 'Qh', '6s'], [12, 10, 958], 2.22],
    [['6c', '8d', '9c', '2h', 'Th', '2s', '9s'], [0, 100.5, 789], 10.15],
    [['6c', '6d', '9c', '7d', 'Jh', 'Ah', '6s'], [946, 0, 44], 95.56],
    [['9c', 'Jc', '5d', '9d', 'Kd', '4h'], [29525, 293, 15429], 65.48],
    [['3c', '4c', '6c', '6d', 'Kd', '7h'], [6778, 2043.5, 34675], 19.37],
    [['Jc', '3d', 'Ac', '9d', '7h', 'Qh'], [12940, 1231.5, 30137], 31.12],
    [['6c', 'Kd', '3c', 'Qc', '5h', '7h'], [16509, 476, 28079], 37.3],
    [['Tc', '9d', '2c', 'Ac', 'Qh', '7s'], [12770, 378, 32014], 28.87],
    [['5c', '8d', '8c', 'Tc', '3h', 'Jh'], [25394, 684, 18778], 57.26],
    [['Tc', 'Qc', 'Ad', 'Jh', '3s'], [544829, 15937, 493487], 52.4],
    [['8c', '6d', '9c', '5d', '3h'], [376668, 10278, 672966], 36.16],
    [['5c', '9d', '9c', 'Qc', 'Jh'], [598361, 32985, 405859], 58.99],
    [['7c', 'Kc', '7d', 'Jd', '6h'], [713844, 4323, 347700], 67.11],
    [['5c', '7d', 'Ac', '5h', '7h'], [877485, 6333.5, 180038], 82.59],
    [['Kc', 'Jd', '2c', '4c', '2d'], [537561, 21002.5, 490624], 52.19],
]


@pytest.mark.parametrize('cards, expected, expected_pct', ps_eval)
def test_evaluate_all_hands(cards, expected, expected_pct):
    cards = cards_to_int(cards)
    result = evaluate_all_hands(cards)
    assert list(result) == expected
    ev_pct = results_to_ev(result)
    assert round(ev_pct * 100, 2) == expected_pct
