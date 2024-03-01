from poker_eval_faster import create_deck_wrapper  # Asegúrate de importar correctamente
import numpy as np

import pytest

DEAD_CARDS = [[], [1, 2, 3], [10, 20, 30, 40, 50], [1, 3, 5, 8, 12, 20, 32], [8, 9]]


def create_expected_deck(dead_card):
    deck = list(range(1, 53))
    for i in dead_card:
        deck.remove(i)
    return deck


@pytest.mark.parametrize('dead_cards', DEAD_CARDS)
def test_create_deck_with_no_dead_cards(dead_cards):
    expected_deck_length = 52 - len(dead_cards)  # Esperamos todas las cartas si no hay cartas muertas
    dead_cards = np.array(dead_cards, dtype="int32")
    result = create_deck_wrapper(dead_cards)
    assert len(result) == expected_deck_length
    deck = create_expected_deck(dead_cards)
    assert len(deck) == expected_deck_length
    for i, card in enumerate(result):
        assert card == deck[i], f"Expected {card} to be in {deck[i]}"

