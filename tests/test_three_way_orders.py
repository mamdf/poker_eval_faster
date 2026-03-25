from math import comb

import numpy as np
import pytest

from poker_eval_faster import (
    THREE_WAY_ORDER_LABELS,
    cards_to_int_array,
    evaluate_hands,
    evaluate_three_way_orders,
    evaluate_three_way_orders_c,
)


FULL_BOARD = ["Ac", "Ad", "2h", "3s", "4d"]
KQ_1 = ["Kc", "Qc"]
KQ_2 = ["Kh", "Qh"]
KQ_3 = ["Ks", "Qs"]
KJ_1 = ["Kd", "Jc"]
KJ_2 = ["Kh", "Jd"]
T9_1 = ["Td", "9d"]

THREE_WAY_COMPLETE_BOARD_CASES = {
    "A>B>C": [KQ_1, KJ_1, T9_1],
    "A>C>B": [KQ_1, T9_1, KJ_1],
    "B>A>C": [KJ_1, KQ_1, T9_1],
    "B>C>A": [T9_1, KQ_1, KJ_1],
    "C>A>B": [KJ_1, T9_1, KQ_1],
    "C>B>A": [T9_1, KJ_1, KQ_1],
    "A=B>C": [KQ_1, KQ_2, T9_1],
    "A=C>B": [KQ_1, T9_1, KQ_2],
    "B=C>A": [T9_1, KQ_1, KQ_2],
    "A>B=C": [KQ_1, KJ_1, KJ_2],
    "B>A=C": [KJ_1, KQ_1, KJ_2],
    "C>A=B": [KJ_1, KJ_2, KQ_1],
    "A=B=C": [KQ_1, KQ_2, KQ_3],
}


def test_evaluate_three_way_orders_c_import_and_shape():
    hands = cards_to_int_array(["Kc", "Qc", "Kd", "Jc", "Td", "9d"])
    board = cards_to_int_array(FULL_BOARD)

    result = evaluate_three_way_orders_c(hands, board)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(THREE_WAY_ORDER_LABELS),)


@pytest.mark.parametrize("expected_label", THREE_WAY_ORDER_LABELS)
def test_evaluate_three_way_orders_matches_each_weak_order(expected_label):
    hands = THREE_WAY_COMPLETE_BOARD_CASES[expected_label]

    result = evaluate_three_way_orders(hands, FULL_BOARD)

    assert result.total == 1
    assert sum(result.order_counts) == 1
    assert result.as_dict()[expected_label] == 1

    for label in THREE_WAY_ORDER_LABELS:
        expected = 1 if label == expected_label else 0
        assert result.as_dict()[label] == expected


@pytest.mark.parametrize("expected_label", THREE_WAY_ORDER_LABELS)
def test_evaluate_three_way_orders_equities_match_evaluate_hands(expected_label):
    hands = THREE_WAY_COMPLETE_BOARD_CASES[expected_label]

    result = evaluate_three_way_orders(hands, FULL_BOARD)
    equities = evaluate_hands(hands, FULL_BOARD)

    assert result.equities == pytest.approx(tuple(equities), abs=1e-12)


def test_evaluate_three_way_orders_first_place_counts_split_two_and_three_way_ties():
    two_way_tie = evaluate_three_way_orders(THREE_WAY_COMPLETE_BOARD_CASES["A=B>C"], FULL_BOARD)
    triple_tie = evaluate_three_way_orders(THREE_WAY_COMPLETE_BOARD_CASES["A=B=C"], FULL_BOARD)

    top_a = two_way_tie.first_place_counts(0)
    top_b = two_way_tie.first_place_counts(1)
    top_c = two_way_tie.first_place_counts(2)
    assert top_a.wins == 0
    assert top_a.two_way_ties == 1
    assert top_a.three_way_ties == 0
    assert top_a.equity == pytest.approx(0.5)
    assert top_b.two_way_ties == 1
    assert top_c.ties == 0

    triple_top = triple_tie.first_place_counts(0)
    assert triple_top.wins == 0
    assert triple_top.two_way_ties == 0
    assert triple_top.three_way_ties == 1
    assert triple_top.equity == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    ("hands", "board", "expected_total"),
    [
        ([["As", "Ks"], ["Qh", "Jh"], ["9c", "9d"]], [], comb(46, 5)),
        ([["As", "Ks"], ["Qh", "Jh"], ["9c", "9d"]], ["2c", "7d", "Th"], comb(43, 2)),
        ([["As", "Ks"], ["Qh", "Jh"], ["9c", "9d"]], ["2c", "7d", "Th", "4s"], 42),
        ([["As", "Ks"], ["Qh", "Jh"], ["9c", "9d"]], ["2c", "7d", "Th", "4s", "3h"], 1),
    ],
)
def test_evaluate_three_way_orders_total_matches_expected_runouts(hands, board, expected_total):
    result = evaluate_three_way_orders(hands, board)

    assert result.total == expected_total
    assert sum(result.order_counts) == expected_total


@pytest.mark.parametrize(
    ("hands", "board"),
    [
        ([["As", "Ks"], ["Qh", "Jh"], ["9c", "9d"]], []),
        ([["As", "Ks"], ["Qh", "Jh"], ["9c", "9d"]], ["2c", "7d", "Th"]),
        ([["As", "Ks"], ["Qh", "Jh"], ["9c", "9d"]], ["2c", "7d", "Th", "4s"]),
        ([["As", "Ks"], ["Qh", "Jh"], ["9c", "9d"]], ["2c", "7d", "Th", "4s", "3h"]),
    ],
)
def test_evaluate_three_way_orders_equities_match_multi_hand_evaluator(hands, board):
    result = evaluate_three_way_orders(hands, board)
    equities = evaluate_hands(hands, board)

    assert result.equities == pytest.approx(tuple(equities), abs=1e-12)
    for idx, equity in enumerate(equities):
        assert result.first_place_counts(idx).equity == pytest.approx(equity, abs=1e-12)


def test_evaluate_three_way_orders_rejects_duplicate_cards():
    result = evaluate_three_way_orders(
        [["As", "Ah"], ["As", "Kd"], ["Qc", "Qd"]],
        ["2c", "7d", "Th"],
    )

    assert result.total == 0
    assert result.order_counts == (0,) * len(THREE_WAY_ORDER_LABELS)


def test_evaluate_three_way_orders_supports_integer_inputs():
    hands = [
        cards_to_int_array(KQ_1),
        cards_to_int_array(KJ_1),
        cards_to_int_array(T9_1),
    ]
    board = cards_to_int_array(FULL_BOARD)

    result = evaluate_three_way_orders(hands, board)

    assert result.total == 1
    assert result.as_dict()["A>B>C"] == 1
