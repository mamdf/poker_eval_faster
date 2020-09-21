from twoplustwo_eval import evaluate_all_hands, cards_to_int
import pytest

ps_river = [[['Ac', 'Ad', 'Ah', 'As', 'Ts', 'Ks', 'Qs'], [946.0, 0.0, 44.0]],
            [['2c', '2d', '5d', '8c', '9c', 'Tc', 'Jc'], [666.0, 0.0, 324.0]],
            [['Ac', 'Kc', '4c', '4d', '4h', '4s', 'Qc'], [861.0, 64.5, 0.0]],
            [['Jc', 'Tc', '3c', '3d', '5c', '5d', '9s'], [226.0, 34.5, 695.0]]]


@pytest.mark.parametrize('cards, expected', ps_river)
def test_evaluate_all_hands(cards, expected):
    cards = cards_to_int(cards)
    assert list(evaluate_all_hands(cards)) == expected
