import numpy as np
import pytest

from poker_eval_faster import canonical_combos, evaluate_ranges, cards_to_int_array
from poker_eval_faster.main import (
    SUITS_STR,
    _canonical_preflop_matchup,
    _preflop_canonical_counts,
    _preflop_range_matchup_profile,
    _range_combo_ids_from_string,
    int_to_cards,
    parse_range_notation,
)


def _combo_id(cards):
    combo = sorted(cards_to_int_array(cards).tolist())
    combos = canonical_combos()
    matches = ((combos[:, 0] == combo[0]) & (combos[:, 1] == combo[1])).nonzero()[0]
    return int(matches[0])

@pytest.mark.parametrize(
    "hero, villain, board,esperado",
    [
        ("AA", "AK+", [], 0.9184),
        ("AK+", "AA", [], 0.0816),
        ("AKs,AKo", "TT+", [], 0.3781),
        ("TT+", "AKs,AKo", [], 0.6219),
        ("99+,AQs+,AQo+", "JJ-77,AQs-A9s,KJs+,QJs,AQo-AJo,KQo", [], 0.6466),
        ("A2s,K2s,Q2s,T6s,95s,84s,62s,52s", "Q5o-Q2o,J9o-J7o,T7o-T6o,96o-95o,85o-84o,74o,63o,53o", [], 0.5084),
        ("A2s+", "K9o+", ["Ac", "Kc", "Qc", "6c", "2c"], 0.3607),
        ("A2s+", "K9s+", ["Ac", "Kc", "Qc", "6c", "2c"], 0.5),
        ("AcQd", "KK+,AK+", [], 0.2358),
        ("KK+,AK+", "AcQd", [], 0.7642),
    ]
)
def test_evaluate_ranges_parametrizado(hero, villain, board, esperado):
    eq = evaluate_ranges(hero, villain, board)
    assert eq == pytest.approx(esperado, abs=1e-4)


def test_evaluate_ranges_iterables():
    # dos combos fijos vs dos combos fijos
    hero = [(cards_to_int_array(['As','Ah'])[0], cards_to_int_array(['As','Ah'])[1])]
    villain = [(cards_to_int_array(['Kd','Kh'])[0], cards_to_int_array(['Kd','Kh'])[1])]
    eq = evaluate_ranges(hero, villain, [])
    assert eq == pytest.approx(0.8195, abs=1e-4)

def _combos_to_cardsets(combos):
    out = set()
    for a, b in combos:
        ca, cb = int_to_cards([a, b])
        out.add(tuple(sorted([ca, cb])))
    return out


def test_parse_counts_and_membership():
    # ATs+ -> ATs, AJs, AQs, AKs (4 ranks * 4 suits = 16 combos)
    combos = parse_range_notation("ATs+")
    s = _combos_to_cardsets(combos)
    assert len(s) == 16
    for r in ["T", "J", "Q", "K"]:
        for suit in SUITS_STR:
            expect = tuple(sorted(["A" + suit, r + suit]))
            assert expect in s

    # K9o+ -> K9o, KTo, KJo, KQo (4 ranks * 12 offsuits = 48)
    combos = parse_range_notation("K9o+")
    s = _combos_to_cardsets(combos)
    assert len(s) == 48
    for r in ["9", "T", "J", "Q"]:
        # ensure suited pairs NOT included
        for suit in SUITS_STR:
            not_suited = tuple(sorted(["K" + suit, r + suit]))
            assert not_suited not in s

    # TT+ -> TT, JJ, QQ, KK, AA (5 ranks * 6 combos = 30)
    combos = parse_range_notation("TT+")
    s = _combos_to_cardsets(combos)
    assert len(s) == 30
    # AA must be present: 6 combos
    count_AA = sum(1 for a, b in s if a[0] == "A" and b[0] == "A")
    assert count_AA == 6

    # A2+ -> A2..AK both suited and offsuit (12 ranks * 16 combos = 192)
    combos = parse_range_notation("A2+")
    s = _combos_to_cardsets(combos)
    assert len(s) == 192

    # AK -> exactly 16 combos (4 suited + 12 offsuit)
    combos = parse_range_notation("AK")
    s = _combos_to_cardsets(combos)
    assert len(s) == 16

def test_suffix_order_variants():
    s1 = _combos_to_cardsets(parse_range_notation("A2s+"))
    s2 = _combos_to_cardsets(parse_range_notation("A2+S"))
    s3 = _combos_to_cardsets(parse_range_notation("A2S+"))
    assert s1 == s2 == s3


def test_dash_notation_ranges():
    # 66-99 -> 4 ranks * 6 combos = 24
    s = _combos_to_cardsets(parse_range_notation("66-99"))
    assert len(s) == 24
    # ATs-A2s -> 9 ranks (2..T) * 4 suited = 36 (A2s..ATs)
    s = _combos_to_cardsets(parse_range_notation("ATs-A2s"))
    assert len(s) == 36
    # ATs-AKs -> 4 ranks (T,J,Q,K) * 4 suited = 16
    s = _combos_to_cardsets(parse_range_notation("ATs-AKs"))
    assert len(s) == 16
    # K9-KQ -> 4 ranks (9,T,J,Q) * 16 (suited+offsuit) = 64
    s = _combos_to_cardsets(parse_range_notation("K9-KQ"))
    assert len(s) == 64

def test_parse_plus_notations():
    from poker_eval_faster.main import parse_range_notation
    combos = parse_range_notation("A2s+")
    assert len(combos) > 0
    combos = parse_range_notation("K9o+")
    assert len(combos) > 0
    combos = parse_range_notation("TT+")
    assert len(combos) > 0


def test_preflop_canonical_matchup_reuses_suit_isomorphisms():
    first_key = _canonical_preflop_matchup(_combo_id(["As", "Kd"]), _combo_id(["Qc", "Jh"]))
    second_key = _canonical_preflop_matchup(_combo_id(["Ah", "Kc"]), _combo_id(["Qd", "Js"]))

    assert first_key == second_key
    assert _preflop_canonical_counts(first_key) == _preflop_canonical_counts(second_key)


def test_preflop_range_matchup_profile_groups_duplicate_pairs():
    hero_ids = _range_combo_ids_from_string("99+,AQs+,AQo+")
    villain_ids = _range_combo_ids_from_string("JJ-77,AQs-A9s,KJs+,QJs,AQo-AJo,KQo")
    profile = _preflop_range_matchup_profile(hero_ids, villain_ids)
    legal_pairs = sum(multiplicity for _, multiplicity in profile)

    assert legal_pairs == 5390
    assert len(profile) < legal_pairs

