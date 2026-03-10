import pytest

from poker_eval_faster import evaluate_heads_up_counts
from tests.testdata import load_cases


HEADS_UP_COUNT_CASES = load_cases("heads_up_counts_snapshot.json")


@pytest.mark.parametrize("case", HEADS_UP_COUNT_CASES, ids=lambda case: case["id"])
def test_evaluate_heads_up_counts_from_json(case):
    counts = evaluate_heads_up_counts(case["hero"], case["villain"], case["board"])
    expected = case["expected"]

    assert counts.wins == expected["wins"]
    assert counts.ties == expected["ties"]
    assert counts.total == expected["total"]
    assert counts.equity == pytest.approx(expected["equity"], abs=1e-12)
