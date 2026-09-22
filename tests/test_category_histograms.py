from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from math import comb
import random

import pytest

from poker_eval_faster import (
    cards_to_int_array, evaluate_rank, ranking_to_category, river_category_histograms,
)
from poker_eval_faster.main import DECK, RANKING


def brute_force(hero, board, available):
    hero_counts = dict.fromkeys(RANKING[1:], 0)
    opponent_counts = dict.fromkeys(RANKING[1:], 0)
    for runout in combinations(available, 5 - len(board)):
        final = [*board, *runout]
        hero_counts[ranking_to_category(evaluate_rank(final, hero))[1]] += 1
        remaining = [card for card in available if card not in runout]
        for opponent in combinations(remaining, 2):
            opponent_counts[ranking_to_category(evaluate_rank(final, opponent))[1]] += 1
    return hero_counts, opponent_counts


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("seed", range(5))
def test_histograms_match_independent_enumeration(size, seed):
    deck = list(DECK)
    random.Random(seed).shuffle(deck)
    hero, board = deck[:2], deck[2:2 + size]
    available = deck[2 + size:10 + size]
    dead = deck[10 + size:]
    result = river_category_histograms(hero, board, dead)
    expected_hero, expected_opponent = brute_force(hero, board, available)
    assert result["hero"] == expected_hero
    assert result["opponent"] == expected_opponent
    assert result["hero_total"] == sum(expected_hero.values()) == comb(8, 5 - size)
    assert result["opponent_total"] == sum(expected_opponent.values())
    assert result["opponent_total"] == comb(8, 5 - size) * comb(8 - (5 - size), 2)


@pytest.mark.parametrize("size,hero_total,opponent_total", [
    (3, 1081, 1070190), (4, 46, 45540), (5, 1, 990),
])
def test_full_deck_totals_and_integer_inputs(size, hero_total, opponent_total):
    hero, board = ["Ac", "Tc"], ["9c", "2d", "As", "2h", "7h"][:size]
    result = river_category_histograms(hero, board)
    assert result["mode"] == "exact"
    assert result["opponent_scope"] == "one_random_opponent"
    assert result["hero_total"] == sum(result["hero"].values()) == hero_total
    assert result["opponent_total"] == sum(result["opponent"].values()) == opponent_total
    assert result == river_category_histograms(cards_to_int_array(hero), cards_to_int_array(board))


def test_royal_flush_board_has_only_straight_flushes():
    result = river_category_histograms(["2c", "3d"], ["Ah", "Kh", "Qh", "Jh", "Th"])
    assert result["hero"]["STRAIGHT_FLUSH"] == result["hero_total"] == 1
    assert result["opponent"]["STRAIGHT_FLUSH"] == result["opponent_total"] == 990
    assert all(value == 0 for key, value in result["opponent"].items() if key != "STRAIGHT_FLUSH")


def test_full_river_matches_enumeration():
    hero, board = ["Ac", "Tc"], ["9c", "2d", "As", "2h", "7h"]
    expected = brute_force(hero, board, [c for c in DECK if c not in hero + board])
    result = river_category_histograms(hero, board)
    assert (result["hero"], result["opponent"]) == expected


def test_parallel_calls_have_independent_counts():
    boards = [["9c", "2d", "As", "2h", "7h"][:size] for size in (3, 4, 5)] * 3
    def evaluate(board):
        return river_category_histograms(["Ac", "Tc"], board)
    expected = list(map(evaluate, boards))
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(evaluate, boards)) == expected


@pytest.mark.parametrize("hero,board,dead", [
    ([1], [2, 3, 4], []),
    ([1, 2], [], []),
    ([1, 2], [3, 4], []),
    ([1, 2], [3, 4, 5, 6, 7, 8], []),
    ([1, 1], [3, 4, 5], []),
    ([1, 2], [2, 4, 5], []),
    ([1, 2], [3, 4, 5], [1]),
    ([0, 2], [3, 4, 5], []),
    ([1, 53], [3, 4, 5], []),
    ([1, 2], [3, 4, 5], [0]),
    ([1, 2], [3, 4, 5], [6, 6]),
    ([1, 2], [3, 4, 5], list(range(6, 50))),
])
def test_invalid_inputs(hero, board, dead):
    with pytest.raises(ValueError):
        river_category_histograms(hero, board, dead)
