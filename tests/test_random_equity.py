from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from itertools import combinations
from math import log, sqrt
import random

import numpy as np
import pytest

from poker_eval_faster import (
    RandomEquityEstimate,
    cards_to_int_array,
    estimate_equity_vs_random,
    evaluate_one_hand_vs_all,
    evaluate_one_hand_vs_two_random,
    evaluate_rank,
)
from poker_eval_faster.main import DECK


HERO = ["Ac", "Tc"]
BOARD = ["9c", "2d", "As", "2h", "7h"]


def exact_reduced_deck(hero, board, available, opponents):
    """Enumerate legal deals independently, including each split-pot size."""
    outcomes = [0] * (opponents + 2)
    for runout in combinations(available, 5 - len(board)):
        final = [*board, *runout]
        hero_rank = evaluate_rank(final, hero)
        deck = [c for c in available if c not in runout]
        ranks = {hand: evaluate_rank(final, hand) for hand in combinations(deck, 2)}

        def visit(remaining, left, winners):
            if not left:
                outcomes[winners] += 1
                return
            for hand in combinations(remaining, 2):
                rank = ranks[hand]
                next_winners = 0 if not winners or rank > hero_rank else winners + (rank == hero_rank)
                visit([c for c in remaining if c not in hand], left - 1, next_winners)

        visit(deck, opponents, 1)
    total = sum(outcomes)
    equity = sum(count / k for k, count in enumerate(outcomes) if k) / total
    return outcomes, equity


def pcg32_reference(seed):
    """Python integer reference, with explicit 32/64-bit wrapping."""
    mask64 = (1 << 64) - 1
    mask32 = (1 << 32) - 1
    state = 0

    def draw():
        nonlocal state
        old = state
        state = (old * 6364136223846793005 + 109) & mask64
        shifted = (((old >> 18) ^ old) >> 27) & mask32
        rotation = old >> 59
        return ((shifted >> rotation) | (shifted << ((-rotation) & 31))) & mask32

    draw()
    state = (state + seed) & mask64
    draw()
    while True:
        yield draw()


def test_pcg_reference_matches_published_vector():
    # https://www.pcg-random.org/using-pcg-c-basic.html (seed 42, stream 54).
    rng = pcg32_reference(42)
    assert [next(rng) for _ in range(6)] == [
        0xA15C02B7, 0x7B47F409, 0xBA1D3330, 0x83D2F293, 0xBFA4784B, 0xCBED606E,
    ]


# Preflop against nine rivals draws 23 cards, the size of the kernel's threshold array.
@pytest.mark.parametrize("size,opponents,seed", [(0, 9, 7), (3, 7, 0), (4, 1, 42), (5, 9, (1 << 64) - 1)])
def test_kernel_matches_independent_deals(size, opponents, seed):
    board = BOARD[:size]
    dead = ["3c", "4d"]
    base = [c for c in DECK if c not in HERO + board + dead]
    rng = pcg32_reference(seed)
    outcomes = [0] * (opponents + 2)
    samples = 2000
    for _ in range(samples):
        deck = base.copy()
        for i in range(5 - size + 2 * opponents):
            bound = len(deck) - i
            r = next(rng)
            while r < (1 << 32) % bound:
                r = next(rng)
            j = i + r % bound
            deck[i], deck[j] = deck[j], deck[i]
        final = board + deck[:5 - size]
        hero_rank = evaluate_rank(final, HERO)
        rivals = [evaluate_rank(final, deck[i:i + 2])
                  for i in range(5 - size, 5 - size + 2 * opponents, 2)]
        winners = 0 if max(rivals) > hero_rank else 1 + rivals.count(hero_rank)
        outcomes[winners] += 1
    result = estimate_equity_vs_random(HERO, board, opponents, dead_cards=dead, samples=samples, seed=seed)
    assert (result.wins, result.ties, result.losses) == (outcomes[1], sum(outcomes[2:]), outcomes[0])
    assert result.equity == pytest.approx(sum(outcomes[k] / k for k in range(1, len(outcomes))) / samples)


@pytest.mark.parametrize("size", [0, 3, 4, 5])
@pytest.mark.parametrize("opponents", [1, 2, 3])
def test_matches_exact_reduced_deck(size, opponents):
    deck = DECK.copy()
    random.Random(17 + size + opponents).shuffle(deck)
    hero, board = deck[:2], deck[2:2 + size]
    # Enough cards for three rivals and the runout; preflop needs eleven.
    end = 2 + size + max(8, 11 - size)
    available, dead = deck[2 + size:end], deck[end:]
    outcomes, equity = exact_reduced_deck(hero, board, available, opponents)
    result = estimate_equity_vs_random(hero, board, opponents, dead_cards=dead, samples=50_000, seed=123)
    # Hoeffding at alpha=1e-9: conservative, fixed tolerance; no seed hunting.
    radius = sqrt(log(2e9) / (2 * result.samples))
    assert abs(result.equity - equity) < radius
    for actual, expected in zip(
        (result.losses, result.wins, result.ties),
        (outcomes[0], outcomes[1], sum(outcomes[2:])),
    ):
        assert abs(actual / result.samples - expected / sum(outcomes)) < radius


@pytest.mark.parametrize("size", [3, 4, 5])
def test_matches_existing_exact_evaluators(size):
    board = BOARD[:size]
    result = estimate_equity_vs_random(HERO, board, 1, seed=42)
    assert abs(result.equity - evaluate_one_hand_vs_all(HERO, board)) < 0.011
    result = estimate_equity_vs_random(HERO, board, 2, seed=42)
    exact = evaluate_one_hand_vs_two_random(HERO, board)
    for actual, expected in zip((result.wins, result.ties, result.losses), exact):
        assert abs(actual / result.samples - expected / sum(exact)) < 0.011


# Exhaustive heads-up preflop counts: pluribus/icm-calculator hu_preflop_equity_v1.bin.
@pytest.mark.parametrize("hero, equity", [
    (["Ac", "Ad"], 0.8520371330210104), (["7c", "2d"], 0.34583647315344157),
    (["Js", "Ts"], 0.5752785710757826), (["5h", "6d"], 0.39944302303939544),
])
def test_preflop_matches_exact_heads_up(hero, equity):
    result = estimate_equity_vs_random(hero, [], 1, seed=42)
    assert abs(result.equity - equity) < sqrt(log(2e9) / (2 * result.samples))


@pytest.mark.parametrize("opponents", [1, 2, 7, 9])
def test_everyone_plays_the_board(opponents):
    result = estimate_equity_vs_random(["2c", "3d"], ["Ah", "Kh", "Qh", "Jh", "Th"], opponents, seed=1)
    assert result.equity == pytest.approx(1 / (opponents + 1))
    assert (result.wins, result.ties, result.losses) == (0, result.samples, 0)


def test_hero_always_wins():
    result = estimate_equity_vs_random(["Ah", "Kh"], ["Qh", "Jh", "Th"], 9, samples=1000, seed=0)
    assert result.equity == 1
    assert (result.wins, result.ties, result.losses) == (1000, 0, 0)


def test_mixed_two_three_and_four_way_splits():
    hero = ["Tc", "2c"]
    board = ["Ah", "Kd", "Qs", "Jc", "4h"]
    available = ["Ts", "Td", "Th", "3c", "5d", "6h", "7s", "8c", "9d", "2h"]
    dead = [c for c in DECK if c not in hero + board + available]
    outcomes, equity = exact_reduced_deck(hero, board, available, 3)
    assert outcomes[0] == 0 and all(count > 0 for count in outcomes[1:])
    result = estimate_equity_vs_random(hero, board, 3, dead_cards=dead, seed=42)
    assert abs(result.equity - equity) < 0.011
    # A generic "half the ties" calculation would overestimate this equity.
    assert (result.wins + result.ties / 2) / result.samples - result.equity > 0.05


def test_tied_rivals_above_hero_are_losses():
    hero = ["2c", "3d"]
    board = ["Ah", "Kd", "Qs", "Jc", "4h"]
    available = ["Tc", "Td", "Th", "Ts"]
    dead = [c for c in DECK if c not in hero + board + available]
    result = estimate_equity_vs_random(hero, board, 2, dead_cards=dead, samples=1000, seed=0)
    assert result.equity == 0
    assert (result.wins, result.ties, result.losses) == (0, 0, 1000)


def test_seed_encoding_metadata_and_concurrency():
    result = estimate_equity_vs_random(HERO, BOARD[:3], 7)
    assert isinstance(result, RandomEquityEstimate)
    assert result.mode == "monte_carlo"
    assert result.samples == 100_000
    assert result.wins + result.ties + result.losses == result.samples
    assert result.error_bound_95 == pytest.approx(0.004294694083467375)
    with pytest.raises(FrozenInstanceError):
        result.equity = 0.0

    def repeat(_):
        return estimate_equity_vs_random(cards_to_int_array(HERO), cards_to_int_array(BOARD[:3]),
                                        np.int64(7), seed=result.seed)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(other == result for other in pool.map(repeat, range(8)))


def test_one_sample_and_smallest_legal_deck():
    hero, board = HERO, BOARD[:3]
    available = ["2c", "3c", "4c", "5c"]
    dead = [c for c in DECK if c not in hero + board + available]
    result = estimate_equity_vs_random(hero, board, 1, dead_cards=dead, samples=1, seed=0)
    assert result.wins + result.ties + result.losses == 1
    assert result.equity in (0, 0.5, 1)
    assert result.error_bound_95 == 1


@pytest.mark.parametrize("kwargs", [
    {"num_opponents": 0}, {"num_opponents": 10}, {"num_opponents": 1.5},
    {"num_opponents": True}, {"samples": 0}, {"samples": -1},
    {"samples": 2.5}, {"samples": True}, {"samples": 1 << 63},
    {"seed": -1}, {"seed": 1 << 64}, {"seed": 1.2}, {"seed": False},
    {"hero": ["Ac"]}, {"hero": ["Ac", "Ac"]}, {"hero": ["Ac", "As"]},
    {"hero": ["Xc", "Tc"]}, {"hero": [0, 2]}, {"hero": [1, 53]},
    {"hero": [1.5, 2]}, {"hero": [True, 2]}, {"hero": [1 << 32, 2]},
    {"board": [], "current_board": True}, {"board": BOARD[:2]}, {"board": BOARD + ["3c"]},
    {"board": ["9c", "9c", "As"]}, {"dead_cards": ["Ac"]},
    {"dead_cards": ["3c", "3c"]}, {"dead_cards": ["9c"]},
    {"dead_cards": [c for c in DECK if c not in HERO + BOARD]},
])
def test_invalid_inputs(kwargs):
    args = dict(hero=HERO, board=BOARD[:3], num_opponents=7, samples=10, seed=0)
    args.update(kwargs)
    with pytest.raises(ValueError):
        estimate_equity_vs_random(**args)
