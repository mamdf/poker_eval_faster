import pytest

from poker_eval_faster import evaluate_three_way_orders, evaluate_three_way_ranges
from tests.testdata import load_cases


THREE_WAY_POKERSTOVE_CASES = load_cases("three_way_pokerstove_snapshot.json")


def _tie_share(result, player_idx: int) -> float:
    counts = result.first_place_counts(player_idx)
    return (counts.two_way_ties / 2.0) + (counts.three_way_ties / 3.0)


@pytest.mark.parametrize("case", THREE_WAY_POKERSTOVE_CASES, ids=lambda case: case["id"])
def test_three_way_results_match_pokerstove_snapshot(case):
    if case["kind"] == "hands":
        result = evaluate_three_way_orders(case["hands"], case["board"])
    elif case["kind"] == "ranges":
        result = evaluate_three_way_ranges(*case["ranges"])
    else:
        raise AssertionError(f"Unsupported fixture kind: {case['kind']!r}")

    expected = case["expected"]

    assert result.total == expected["total"]
    assert len(expected["players"]) == 3

    for player_idx, player_expected in enumerate(expected["players"]):
        first_place = result.first_place_counts(player_idx)
        assert first_place.wins == player_expected["wins"]
        assert _tie_share(result, player_idx) == pytest.approx(player_expected["tie_share"], abs=1e-6)
        assert result.equities[player_idx] == pytest.approx(player_expected["equity"], abs=1e-12)
