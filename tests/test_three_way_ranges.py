from math import comb

import numpy as np
import pytest

from poker_eval_faster import (
    THREE_WAY_ORDER_LABELS,
    canonical_combos,
    cards_to_int_array,
    evaluate_three_way_orders,
    evaluate_three_way_orders_c,
    evaluate_three_way_ranges,
)
from poker_eval_faster.main import (
    _canonical_preflop_three_way_matchup,
    _preflop_three_way_canonical_counts,
    _preflop_three_way_range_matchup_profile,
    _range_combo_ids_from_string,
    parse_range_notation,
)


def _combo_id(cards):
    combo = sorted(cards_to_int_array(cards).tolist())
    combos = canonical_combos()
    matches = ((combos[:, 0] == combo[0]) & (combos[:, 1] == combo[1])).nonzero()[0]
    return int(matches[0])


def _range_combos(range_input):
    if isinstance(range_input, str):
        combos = parse_range_notation(range_input)
    else:
        combos = list(range_input)
    return [tuple(sorted((int(first), int(second)))) for first, second in combos]


def _explicit_three_way_aggregate(range_a, range_b, range_c):
    combos_a = _range_combos(range_a)
    combos_b = _range_combos(range_b)
    combos_c = _range_combos(range_c)
    total_counts = [0] * len(THREE_WAY_ORDER_LABELS)
    legal_triples = 0

    for combo_a in combos_a:
        cards_a = set(combo_a)
        for combo_b in combos_b:
            if cards_a.intersection(combo_b):
                continue
            cards_ab = cards_a.union(combo_b)
            for combo_c in combos_c:
                if cards_ab.intersection(combo_c):
                    continue
                legal_triples += 1
                result = evaluate_three_way_orders([combo_a, combo_b, combo_c])
                for idx, count in enumerate(result.order_counts):
                    total_counts[idx] += count

    return tuple(total_counts), legal_triples


def test_evaluate_three_way_orders_preflop_matches_raw_kernel():
    hands = [["As", "Ks"], ["Qh", "Jh"], ["9c", "9d"]]
    raw_hands = cards_to_int_array(["As", "Ks", "Qh", "Jh", "9c", "9d"])
    raw_counts = evaluate_three_way_orders_c(raw_hands, np.array([], dtype="int32"))

    result = evaluate_three_way_orders(hands)

    assert result.total == comb(46, 5)
    assert result.order_counts == tuple(int(value) for value in raw_counts.tolist())


def test_preflop_three_way_canonical_matchup_reuses_suit_isomorphisms():
    first_key = _canonical_preflop_three_way_matchup(
        _combo_id(["As", "Kd"]),
        _combo_id(["Qc", "Jh"]),
        _combo_id(["9s", "8h"]),
    )
    second_key = _canonical_preflop_three_way_matchup(
        _combo_id(["Ah", "Kc"]),
        _combo_id(["Qs", "Jd"]),
        _combo_id(["9h", "8d"]),
    )

    assert first_key == second_key
    assert _preflop_three_way_canonical_counts(first_key) == _preflop_three_way_canonical_counts(second_key)


def test_preflop_three_way_range_matchup_profile_groups_duplicate_triples():
    range_a_ids = _range_combo_ids_from_string("AA")
    range_b_ids = _range_combo_ids_from_string("KK")
    range_c_ids = _range_combo_ids_from_string("QQ")
    profile = _preflop_three_way_range_matchup_profile(range_a_ids, range_b_ids, range_c_ids)
    legal_triples = sum(multiplicity for _, multiplicity in profile)

    assert legal_triples > 0
    assert len(profile) < legal_triples


def test_evaluate_three_way_ranges_matches_explicit_combo_aggregation():
    range_a = "AcKc,AdKd"
    range_b = "2c2d,3c3d"
    range_c = "5s6s,7s8s"
    expected_counts, legal_triples = _explicit_three_way_aggregate(range_a, range_b, range_c)

    result = evaluate_three_way_ranges(range_a, range_b, range_c)

    assert result.order_counts == expected_counts
    assert result.total == legal_triples * comb(46, 5)


def test_evaluate_three_way_ranges_supports_mixed_input_types():
    range_a = "AcKc,AdKd"
    range_b = parse_range_notation("2c2d,3c3d")
    range_c = tuple(parse_range_notation("5s6s,7s8s"))

    result_from_strings = evaluate_three_way_ranges("AcKc,AdKd", "2c2d,3c3d", "5s6s,7s8s")
    mixed_result = evaluate_three_way_ranges(range_a, range_b, range_c)

    assert mixed_result.order_counts == result_from_strings.order_counts
    assert mixed_result.equities == pytest.approx(result_from_strings.equities, abs=1e-12)


def test_evaluate_three_way_ranges_returns_zero_when_no_legal_triples_exist():
    result = evaluate_three_way_ranges("AcKc", "AcQd", "AcJd")

    assert result.total == 0
    assert result.order_counts == (0,) * len(THREE_WAY_ORDER_LABELS)
