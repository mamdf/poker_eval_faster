from math import comb

import pytest

from poker_eval_faster import (
    HU_LOOKUP_TOTAL_RUNOUTS,
    build_hu_preflop_lookup,
    canonical_combo_masks,
    canonical_combos,
    cards_to_int_array,
    combo_id_to_cards,
    combo_id_to_str,
    evaluate_heads_up_counts,
    legal_pair_count,
    packed_pair_index,
    read_hu_preflop_lookup,
    triangular_pair_count,
    write_hu_preflop_lookup,
)


def _combo_id(cards):
    combo = sorted(cards_to_int_array(cards).tolist())
    combos = canonical_combos()
    matches = ((combos[:, 0] == combo[0]) & (combos[:, 1] == combo[1])).nonzero()[0]
    return int(matches[0])


def test_legal_pair_count_matches_full_preflop_combinatorics():
    assert triangular_pair_count(1326) == comb(1326, 2)
    assert legal_pair_count(canonical_combo_masks()) == comb(52, 4) * 3


def test_combo_id_string_helpers_match_artifact_helpers():
    aa_idx = _combo_id(["As", "Ah"])
    kk_idx = _combo_id(["Ks", "Kh"])
    artifact = build_hu_preflop_lookup(combo_indices=[aa_idx, kk_idx])

    assert combo_id_to_cards(aa_idx) == ["Ah", "As"]
    assert combo_id_to_str(aa_idx) == "AhAs"
    assert artifact.combo_id_cards(aa_idx) == ["Ah", "As"]
    assert artifact.combo_id_str(aa_idx) == "AhAs"
    assert artifact.local_combo_str(artifact.local_combo_index(kk_idx)) == "KhKs"


def test_hu_lookup_artifact_subset_uses_sentinel_for_overlap():
    aa_idx = _combo_id(["As", "Ah"])
    kk_idx = _combo_id(["Ks", "Kh"])
    ak_idx = _combo_id(["As", "Kd"])

    artifact = build_hu_preflop_lookup(combo_indices=[aa_idx, kk_idx, ak_idx])

    assert artifact.header.combo_count == 3
    assert artifact.header.entry_count == 3
    assert artifact.header.legal_count == 2
    assert artifact.header.total_runouts == HU_LOOKUP_TOTAL_RUNOUTS
    assert artifact.combos["combo_id"].tolist() == sorted([aa_idx, kk_idx, ak_idx])

    overlap_pair_idx = packed_pair_index(0, 2, artifact.header.combo_count)
    assert int(artifact.entries["win"][overlap_pair_idx]) == artifact.header.sentinel
    assert int(artifact.entries["tie"][overlap_pair_idx]) == artifact.header.sentinel
    assert artifact.lookup(0, 2) is None


def test_hu_lookup_artifact_roundtrip_and_ordered_lookup(tmp_path):
    aa_idx = _combo_id(["As", "Ah"])
    kk_idx = _combo_id(["Ks", "Kh"])
    qj_idx = _combo_id(["Qs", "Jd"])
    path = tmp_path / "hu_preflop_subset.bin"

    summary = write_hu_preflop_lookup(path, combo_indices=[aa_idx, kk_idx, qj_idx])
    artifact = read_hu_preflop_lookup(path)

    assert summary.path == path
    assert artifact.header.combo_count == 3
    assert artifact.header.legal_count == 3

    expected = evaluate_heads_up_counts(["As", "Ah"], ["Ks", "Kh"], [])
    stored = artifact.lookup_combo_ids_ordered(aa_idx, kk_idx)
    reverse = artifact.lookup_combo_ids_ordered(kk_idx, aa_idx)

    assert stored == expected
    assert reverse is not None
    assert reverse.wins == expected.losses
    assert reverse.ties == expected.ties
    assert reverse.total == expected.total


@pytest.mark.parametrize(
    ("hero", "villain"),
    [
        (["As", "Ah"], ["Ks", "Kh"]),
        (["Ac", "Kd"], ["Qs", "Jh"]),
        (["9c", "8c"], ["Td", "Ts"]),
    ],
)
def test_hu_lookup_artifact_matches_direct_counts(hero, villain):
    hero_idx = _combo_id(hero)
    villain_idx = _combo_id(villain)
    artifact = build_hu_preflop_lookup(combo_indices=[hero_idx, villain_idx])

    direct = evaluate_heads_up_counts(hero, villain, [])
    assert artifact.lookup_combo_ids_ordered(hero_idx, villain_idx) == direct
