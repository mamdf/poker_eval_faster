import importlib.util
from itertools import permutations
from math import comb, log, sqrt
from pathlib import Path

import numpy as np
import pytest

from poker_eval_faster import (
    HAND_CLASSES_169,
    build_three_way_class_lookup,
    class_label_to_id,
    estimate_equity_vs_random,
    evaluate_ranges,
    evaluate_three_way_ranges,
    preflop_class_equities,
    preflop_equity_vs_random,
)
from poker_eval_faster.preflop_tables import (
    PREFLOP_HU_FILE,
    PREFLOP_THREE_WAY_FILE,
    PREFLOP_VS_RANDOM_FILE,
    _heads_up_class_counts,
    _legal_triple_counts,
    _table,
    _three_way_numerators,
    _vs_two_random,
)

ROOT = Path(__file__).resolve().parents[1]
# Exact heads-up equities against one random hand (pluribus icm-calculator table).
EXACT_VS_ONE_RANDOM = {
    "AA": 0.8520371330210104,
    "72o": 0.34583647315344157,
    "JTs": 0.5752785710757826,
    "65o": 0.39944302303939544,
}


def _load_build_script():
    spec = importlib.util.spec_from_file_location("build_preflop_tables", ROOT / "scripts/build_preflop_equity_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tables_use_the_library_class_order():
    assert (HAND_CLASSES_169[0], HAND_CLASSES_169[13], HAND_CLASSES_169[-1]) == ("AA", "AKs", "32o")
    assert _table(PREFLOP_VS_RANDOM_FILE).shape == (169, 9)
    assert _table(PREFLOP_HU_FILE).shape == (169, 169)
    assert _table(PREFLOP_THREE_WAY_FILE).shape == (818805, 3)
    assert _table(PREFLOP_THREE_WAY_FILE).dtype == np.float32


def test_heads_up_class_table_is_exact_and_antisymmetric():
    table = _table(PREFLOP_HU_FILE)
    np.testing.assert_allclose(table + table.T, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.diag(table), 0.5)
    for hero, villain in (("AA", "KK"), ("AKs", "QQ"), ("72o", "32s"), ("JTs", "JTs")):
        equities = preflop_class_equities((hero, villain))
        assert equities[0] == pytest.approx(evaluate_ranges(hero, villain), abs=1e-12)
        assert sum(equities) == pytest.approx(1.0, abs=1e-12)


def test_three_way_table_marks_exactly_the_impossible_triples():
    table = np.asarray(_table(PREFLOP_THREE_WAY_FILE))
    rank_counts = np.zeros((169, 13), dtype=np.int64)
    for class_id, label in enumerate(HAND_CLASSES_169):
        for rank in label[:2]:
            rank_counts[class_id, "23456789TJQKA".index(rank)] += 1
    triples = np.array(
        [(a, b, c) for a in range(169) for b in range(a, 169) for c in range(b, 169)], dtype=np.int64
    )
    impossible = (rank_counts[triples].sum(axis=1) > 4).any(axis=1)

    assert impossible.sum() == 325
    np.testing.assert_array_equal(np.isnan(table[:, 0]), impossible)
    np.testing.assert_allclose(table[~impossible].sum(axis=1), 1.0, rtol=0, atol=1e-5)


@pytest.mark.parametrize("classes", [("AA", "KK", "QQ"), ("AKo", "AA", "KQo"), ("KK", "AA", "KK"), ("72o", "32s", "22")])
def test_three_way_classes_match_exact_ranges_in_caller_order(classes):
    for ordered in permutations(classes):
        expected = evaluate_three_way_ranges(*ordered).equities
        assert preflop_class_equities(ordered) == pytest.approx(expected, abs=1e-6)


def test_impossible_three_way_deal_raises():
    with pytest.raises(ValueError, match="No legal deal"):
        preflop_class_equities(("AA", "AA", "AKs"))


@pytest.mark.parametrize("label, expected", EXACT_VS_ONE_RANDOM.items())
def test_equity_vs_one_random_is_exact(label, expected):
    assert preflop_equity_vs_random(label) == pytest.approx(expected, abs=1e-12)


def test_cards_and_labels_give_the_same_equity():
    assert preflop_equity_vs_random(["Ac", "Ad"]) == preflop_equity_vs_random("AA")
    assert preflop_equity_vs_random(["7c", "2d"], 3) == preflop_equity_vs_random("72o", 3)
    assert preflop_equity_vs_random([52, 48]) == preflop_equity_vs_random("aks")


@pytest.mark.parametrize("label", ["AA", "72o", "JTs", "65o"])
def test_equity_vs_more_randoms_agrees_with_monte_carlo(label):
    hero = {"AA": ["Ac", "Ad"], "72o": ["7c", "2d"], "JTs": ["Js", "Ts"], "65o": ["6h", "5d"]}[label]
    samples = 200_000
    radius = sqrt(log(2 / 1e-9) / (2 * samples))
    for opponents in range(2, 10):
        estimate = estimate_equity_vs_random(hero, [], opponents, samples=samples, seed=opponents).equity
        assert abs(preflop_equity_vs_random(label, opponents) - estimate) <= radius


def test_equity_vs_random_decreases_with_opponents():
    assert np.all(np.diff(_table(PREFLOP_VS_RANDOM_FILE), axis=1) < 0)


def test_derivation_helpers_match_exact_ranges_on_a_small_universe():
    labels = ["AA", "KK", "AKs", "AKo", "72o"]  # Library order, as the lookup builder sorts ids.
    ids = [class_label_to_id(label) for label in labels]
    universe = ",".join(labels)
    num2, total = _heads_up_class_counts(ids)
    num6, runouts = _three_way_numerators(build_three_way_class_lookup(ids, processes=1).entries)
    np.testing.assert_array_equal(_legal_triple_counts(ids) * comb(46, 5), runouts)
    pot6, deals = _vs_two_random(num6, 6 * runouts, len(ids))
    for idx, label in enumerate(labels):
        assert num2[idx].sum() / (2 * total[idx].sum()) == pytest.approx(evaluate_ranges(label, universe), abs=1e-12)
        expected = evaluate_three_way_ranges(label, universe, universe).equities[0]
        assert pot6[idx] / deals[idx] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("hand", ["AK", "AAs", "XYs", ["Ac", "Ac"], ["Ac"], ["Ac", "Kd", "Qh"], ["Zz", "Ac"]])
def test_invalid_hands_raise(hand):
    with pytest.raises(ValueError):
        preflop_equity_vs_random(hand)


@pytest.mark.parametrize("num_opponents", [0, 10, 1.5, True])
def test_invalid_opponent_counts_raise(num_opponents):
    with pytest.raises(ValueError):
        preflop_equity_vs_random("AA", num_opponents)


@pytest.mark.parametrize("classes", ["AAKK", ("AA",), ("AA", "KK", "QQ", "JJ"), ("AA", 12), (["Ac", "Ad"], "KK"), ("AK", "QQ")])
def test_invalid_class_sequences_raise(classes):
    with pytest.raises(ValueError):
        preflop_class_equities(classes)


def test_legacy_class_order_is_a_bijection():
    legacy_label = _load_build_script()._legacy_label
    labels = [legacy_label(index) for index in range(169)]
    assert sorted(labels) == sorted(HAND_CLASSES_169)
    assert [labels[i] for i in (0, 1, 2, 90, 91, 168)] == ["22", "32o", "33", "AA", "32s", "AKs"]
