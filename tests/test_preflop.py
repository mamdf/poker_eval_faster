from twoplustwo_eval.isomorphism_preflop import fix_hole_cards
from twoplustwo_eval.helper import hands_to_int
import pytest

hands = [
    [['7s', 'Kh'], ['Ad', 'Ac'], ['6d', 'Js']],
    [['Ac', 'Ad'], ['Kh', '7s'], ['Js', '6d']],
    [['7h', 'Kc'], ['As', 'Ad'], ['6s', 'Jh']],
    [['7c', 'Kd'], ['Ah', 'As'], ['6h', 'Jc']],
    [['7s', 'Kh'], ['Ad', 'Ac'], ['6c', 'Js']],
    [['Ac', 'Ad'], ['7h', 'Ks'], ['Jh', '6d']],
]


@pytest.mark.parametrize('hand', hands)
def test_fix_hole_cards(hand):
    int_hand = hands_to_int(hand)
    assert fix_hole_cards(int_hand) == [17, 38, 22, 47, 49, 52], hand
