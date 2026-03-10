import pytest
from typing import cast

from poker_eval_faster import evaluate_one_hand_vs_all
from tests.testdata import load_cases


ONE_HAND_VS_ALL_CASES = load_cases("one_hand_vs_all_pokerstove.json")


@pytest.mark.parametrize("case", ONE_HAND_VS_ALL_CASES, ids=lambda case: case["id"])
def test_evaluate_one_hand_vs_all(case):
    result = evaluate_one_hand_vs_all(case["hand"], case["board"])
    assert round(cast(float, result) * 100, 2) == case["expected_equity_pct"]

