from math import comb

import numpy as np
import pytest

from poker_eval_faster import (
    aggregate_heads_up_lookup_by_class,
    build_heads_up_lookup,
    canonical_combos,
    cards_to_int_array,
    combo_to_hand_class,
    evaluate_hands,
    evaluate_heads_up_counts,
)
from poker_eval_faster.experimental import distribution_one_hand_vs_all as experimental_distribution


def _combo_id(cards):
    combo = np.sort(cards_to_int_array(cards))
    combos = canonical_combos()
    matches = np.where((combos[:, 0] == combo[0]) & (combos[:, 1] == combo[1]))[0]
    return int(matches[0])


def test_evaluate_heads_up_counts_matches_equity_wrapper():
    hero = ['Tc', 'Qc']
    villain = ['9c', '8c']
    board = ['Qh', 'Jh', '8s']

    counts = evaluate_heads_up_counts(hero, villain, board)
    equities = evaluate_hands([hero, villain], board)

    assert counts.total == comb(45, 2)
    assert counts.equity == pytest.approx(equities[0], abs=1e-10)


def test_evaluate_heads_up_counts_rejects_duplicate_cards():
    counts = evaluate_heads_up_counts(['Ac', 'Ad'], ['Ac', 'Kd'], [])
    assert counts.total == 0
    assert counts.wins == 0
    assert counts.ties == 0


def test_build_heads_up_lookup_subset_and_overlap_handling():
    aa_idx = _combo_id(['As', 'Ah'])
    kk_idx = _combo_id(['Ks', 'Kh'])
    ak_idx = _combo_id(['As', 'Kd'])

    lookup = build_heads_up_lookup(combo_indices=[aa_idx, kk_idx, ak_idx])
    direct_counts = evaluate_heads_up_counts(['As', 'Ah'], ['Ks', 'Kh'], [])

    assert lookup.total == comb(48, 5)

    aa_vs_kk = lookup.counts_for_ids(aa_idx, kk_idx)
    assert aa_vs_kk == direct_counts

    aa_vs_ak = lookup.counts_for_ids(aa_idx, ak_idx)
    assert aa_vs_ak.total == 0


def test_aggregate_heads_up_lookup_by_class_uses_canonical_labels():
    aa_idx = _combo_id(['As', 'Ah'])
    kk_idx = _combo_id(['Ks', 'Kh'])

    lookup = build_heads_up_lookup(combo_indices=[aa_idx, kk_idx])
    aggregated = aggregate_heads_up_lookup_by_class(lookup)
    direct_counts = evaluate_heads_up_counts(['As', 'Ah'], ['Ks', 'Kh'], [])

    assert combo_to_hand_class(['As', 'Ah']) == 'AA'
    assert combo_to_hand_class(['As', 'Ks']) == 'AKs'
    assert combo_to_hand_class(['As', 'Kd']) == 'AKo'
    assert aggregated[('AA', 'KK')] == direct_counts
    assert aggregated[('KK', 'AA')].wins == direct_counts.losses


def test_experimental_distribution_namespace_keeps_current_behavior():
    result = experimental_distribution(['2s', '6s'], ['Qs', 'Ks', '7c', 'Ah'])
    assert isinstance(result, np.ndarray)
    assert result.shape == (53,)
