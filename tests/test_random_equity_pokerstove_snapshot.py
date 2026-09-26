import json
from math import log, sqrt
from pathlib import Path

import pytest

from poker_eval_faster import estimate_equity_vs_random


SNAPSHOT = json.loads((Path(__file__).parent / "fixtures" / "random_equity_pokerstove_snapshot.json").read_text())


@pytest.mark.parametrize("case", SNAPSHOT["cases"], ids=lambda case: case["id"])
def test_random_equity_matches_independent_pokerstove(case):
    # Hold out a seed not used in the calibration report. Allow for both
    # samplers' uncertainty; exact references have no sampling variance.
    reference = case["reference"]
    samples = 100_000
    inverse_reference = 0 if reference["mode"] == "exact" else 1 / reference["total"]
    # Two-sample Hoeffding bound; union bound over the suite at alpha=1e-9.
    radius = sqrt(log(2 * len(SNAPSHOT["cases"]) / 1e-9) * (1 / samples + inverse_reference) / 2)
    result = estimate_equity_vs_random(case["hero"], case["board"], case["players"] - 1,
                                       dead_cards=case["dead"], samples=samples, seed=777)
    assert abs(result.equity - reference["equity"]) < radius
