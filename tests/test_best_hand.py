from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from poker_eval_faster import BestHand, cards_to_int_array, evaluate_best_hand, evaluate_rank


@pytest.mark.parametrize('board,hand,board_indices,hand_indices', [
    ('7d Qs 9h 5h 9c', 'Tc 7s', (0, 1, 2, 4), (1,)),
    ('7d Qs 9h 5h 9c', 'Jd 7h', (0, 1, 2, 4), (1,)),
    ('Ac Kd Qh Js Tc', 'Ad Kh', (0, 1, 2, 3, 4), ()),
    ('2c 3d 4h 5s Kc', 'Ac Ad', (0, 1, 2, 3), (0,)),
    ('Ac Ad 3h 7s 9c', 'Kh Qd', (0, 1, 4), (0, 1)),
    ('Ac 2d 3h 4s Kc', '5d Qh', (0, 1, 2, 3), (0,)),
    ('Ac Kc Qc Jc Td', '9c 8c', (0, 1, 2, 3), (0,)),
    ('Ac Ad Ah Kc Qc', 'Kd Qd', (0, 1, 2, 3), (0,)),
    ('Ac Ad Ah As Kc', 'Kd Qd', (0, 1, 2, 3, 4), ()),
    ('2c 3d 4h 4s 5c', '6d Kh', (0, 1, 2, 4), (0,)),
])
def test_selected_cards(board, hand, board_indices, hand_indices):
    board, hand = board.split(), hand.split()
    result = evaluate_best_hand(board, hand)
    assert isinstance(result, BestHand)
    assert result.board_indices == board_indices
    assert result.hand_indices == hand_indices
    selected = [board[i] for i in result.board_indices] + [hand[i] for i in result.hand_indices]
    assert len(selected) == 5
    assert result.rank == evaluate_rank(board, hand) == evaluate_rank(selected)


@pytest.mark.parametrize('count', [5, 6, 7])
@pytest.mark.parametrize('encoding', ['strings', 'integers', 'array'])
def test_card_encodings_and_lengths(count, encoding):
    cards = 'Ac Kd Qh Js Tc 2c 3d'.split()[:count]
    if encoding != 'strings':
        cards = cards_to_int_array(cards)
        if encoding == 'integers':
            cards = cards.tolist()
    result = evaluate_best_hand(cards)
    assert result.board_indices == (0, 1, 2, 3, 4)
    assert result.hand_indices == ()
    assert result.rank == evaluate_rank(cards)


def test_frozen_result_and_unchanged_inputs():
    board = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    before = board.copy()
    result = evaluate_best_hand(board)
    np.testing.assert_array_equal(board, before)
    with pytest.raises(FrozenInstanceError):
        result.rank = 0


@pytest.mark.parametrize('cards', [[], [1, 2, 3, 4], list(range(1, 9)), [1, 2, 3, 4, 4], [0, 2, 3, 4, 5], [53, 2, 3, 4, 5]])
def test_invalid_cards(cards):
    with pytest.raises(ValueError):
        evaluate_best_hand(cards)


def test_duplicate_across_board_and_hand():
    with pytest.raises(ValueError, match='distinct'):
        evaluate_best_hand('Ac Kd Qh Js Tc'.split(), ['Ac', '2d'])
