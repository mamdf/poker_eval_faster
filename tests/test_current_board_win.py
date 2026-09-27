"""Current-board events stop at the visible street."""

import pytest

from poker_eval_faster import (
    estimate_equity_vs_random,
    evaluate_one_hand_vs_all,
    evaluate_one_hand_vs_two_random,
)


def test_exact_heads_up_current_board_differs_from_river_runout():
    hero = ["As", "Qs"]
    board = ["2c", "5d", "8h", "Ts"]
    current = evaluate_one_hand_vs_all(hero, board, eq=False, incomplete_board=True)
    river = evaluate_one_hand_vs_all(hero, board, eq=False)
    assert sum(current) > 0 and sum(river) > sum(current)
    assert current[0] / sum(current) != pytest.approx(river[0] / sum(river))


def test_exact_two_random_current_board_counts_all_events():
    hero = ["As", "Qs"]
    board = ["2c", "5d", "8h", "Ts"]
    current = evaluate_one_hand_vs_two_random(hero, board, current_board=True)
    river = evaluate_one_hand_vs_two_random(hero, board)
    assert sum(current) > 0 and sum(river) > sum(current)
    assert all(value >= 0 for value in current)


def test_monte_carlo_current_board_seed_and_dead_cards():
    args = (["As", "Qs"], ["2c", "5d", "8h", "Ts"], 3)
    a = estimate_equity_vs_random(*args, dead_cards=["Ac"], samples=500, seed=19,
                                  current_board=True)
    b = estimate_equity_vs_random(*args, dead_cards=["Ac"], samples=500, seed=19,
                                  current_board=True)
    assert a == b and a.wins + a.ties + a.losses == 500
    assert a != estimate_equity_vs_random(*args, dead_cards=["Ac"], samples=500, seed=19)
