from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from math import comb
import random

import pytest

from poker_eval_faster import evaluate_one_hand_vs_two_random, evaluate_rank
from poker_eval_faster.main import DECK


def brute_force(hero, board, available):
    """Independent ordered rival-pair enumeration; no degree-count formula."""
    counts = [0, 0, 0]
    for runout in combinations(available, 5 - len(board)):
        final = [*board, *runout]
        deck = [c for c in available if c not in runout]
        rank = evaluate_rank(final, hero)
        hands = [
            ((1 << a) | (1 << b), evaluate_rank(final, [deck[a], deck[b]]))
            for a, b in combinations(range(len(deck)), 2)
        ]
        for i, (mask_a, rank_a) in enumerate(hands):
            for mask_b, rank_b in hands[i + 1:]:
                if mask_a & mask_b:
                    continue
                rival = max(rank_a, rank_b)
                # Both seat assignments have the same outcome.
                counts[0 if rank > rival else 1 if rank == rival else 2] += 2
    return tuple(counts)


@pytest.mark.parametrize("size", [3, 4, 5])
@pytest.mark.parametrize("seed", range(5))
def test_reduced_deck_matches_direct_enumeration(size, seed):
    deck = list(DECK)
    random.Random(seed).shuffle(deck)
    hero, board = deck[:2], deck[2:2 + size]
    available = deck[2 + size:12 + size]
    dead = deck[12 + size:]
    actual = evaluate_one_hand_vs_two_random(hero, board, dead)
    assert actual == brute_force(hero, board, available)
    assert sum(actual) == comb(10, 2) * comb(8, 2) * comb(6, 5 - size)


@pytest.mark.parametrize("hero,board", [
    (["Ac", "Tc"], ["9c", "2d", "As", "2h", "7h"]),
    (["As", "Qs"], ["Ah", "Kd", "Jc", "Ts", "2c"]),
    (["2c", "3d"], ["Ah", "Kh", "Qh", "Jh", "Th"]),
])
def test_full_river_matches_direct_enumeration(hero, board):
    available = [c for c in DECK if c not in hero + board]
    actual = evaluate_one_hand_vs_two_random(hero, board)
    assert actual == brute_force(hero, board, available)
    assert sum(actual) == 893970


@pytest.mark.parametrize("size,total", [(3, 966381570), (4, 41122620), (5, 893970)])
def test_full_deck_totals_and_reentrancy(size, total):
    board = ["9c", "2d", "As", "2h", "7h"][:size]
    def evaluate(_):
        return evaluate_one_hand_vs_two_random(["Ac", "Tc"], board)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(evaluate, range(6)))
    assert all(result == results[0] for result in results)
    assert sum(results[0]) == total
    assert all(n >= 0 for n in results[0])


@pytest.mark.parametrize("hero,board,dead", [
    ([1], [2, 3, 4], []),
    ([1, 2], [], []),
    ([1, 2], [3, 4], []),
    ([1, 1], [3, 4, 5], []),
    ([1, 2], [2, 4, 5], []),
    ([1, 2], [3, 4, 5], [1]),
    ([0, 2], [3, 4, 5], []),
    ([1, 53], [3, 4, 5], []),
    ([1, 2], [3, 4, 5], list(range(6, 50))),
])
def test_reject_invalid_inputs(hero, board, dead):
    with pytest.raises(ValueError):
        evaluate_one_hand_vs_two_random(hero, board, dead)


@pytest.mark.parametrize("size", [3, 4])
def test_runout_decomposition(size):
    hero = ["Ac", "Tc"]
    board = ["9c", "2d", "As", "2h"][:size]
    actual = evaluate_one_hand_vs_two_random(hero, board)
    children = [
        evaluate_one_hand_vs_two_random(hero, [*board, card])
        for card in DECK if card not in hero + board
    ]
    # Each unordered flop runout appears twice when choosing turn first.
    assert tuple(sum(c[i] for c in children) for i in range(3)) == tuple(
        count * (5 - size) for count in actual
    )
