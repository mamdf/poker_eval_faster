import numpy as np
import pytest

from poker_eval_faster import evaluate_ranges, cards_to_int_array


def test_evaluate_ranges_simple_pairs_vs_aks():
    # TT explicitado como todos los combos off-suit (placeholder)
    hero = "TT"
    villain = "AsKs"
    eq = evaluate_ranges(hero, villain, [])
    assert 0.0 <= eq <= 1.0


def test_evaluate_ranges_iterables():
    # dos combos fijos vs dos combos fijos
    hero = [(cards_to_int_array(['As','Ah'])[0], cards_to_int_array(['As','Ah'])[1])]
    villain = [(cards_to_int_array(['Kd','Kh'])[0], cards_to_int_array(['Kd','Kh'])[1])]
    eq = evaluate_ranges(hero, villain, [])
    assert 0.0 <= eq <= 1.0


