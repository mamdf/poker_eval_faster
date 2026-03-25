from itertools import permutations
from math import comb

import numpy as np

from poker_eval_faster import (
    HAND_CLASSES_169,
    THREE_WAY_CLASS_LOOKUP_MAGIC,
    THREE_WAY_CLASS_LOOKUP_TOTAL_RUNOUTS,
    THREE_WAY_CLASS_LOOKUP_VERSION,
    build_three_way_class_lookup,
    class_combo_ids,
    class_id_to_label,
    class_label_to_id,
    evaluate_three_way_ranges,
    multiset_triple_count,
    packed_class_triple_index,
    read_three_way_class_lookup,
    unpack_class_triple_index,
    write_three_way_class_lookup,
)


def test_hand_classes_169_roundtrip_and_combo_sizes():
    assert len(HAND_CLASSES_169) == 169
    assert len(set(HAND_CLASSES_169)) == 169
    assert HAND_CLASSES_169[0] == "AA"
    assert HAND_CLASSES_169[-1] == "32o"

    for label in ("AA", "AKs", "AKo", "T9s", "54o", "22"):
        class_id = class_label_to_id(label)
        assert class_id_to_label(class_id) == label

    assert len(class_combo_ids("AA")) == 6
    assert len(class_combo_ids("AKs")) == 4
    assert len(class_combo_ids("AKo")) == 12


def test_multiset_triple_count_matches_combinatorics():
    assert multiset_triple_count(0) == 0
    assert multiset_triple_count(1) == 1
    assert multiset_triple_count(3) == comb(5, 3)
    assert multiset_triple_count(169) == comb(171, 3)


def test_packed_class_triple_index_roundtrips_for_small_subset():
    num_items = 5
    seen = set()
    for first_idx in range(num_items):
        for second_idx in range(first_idx, num_items):
            for third_idx in range(second_idx, num_items):
                packed_idx = packed_class_triple_index(first_idx, second_idx, third_idx, num_items)
                assert packed_idx not in seen
                seen.add(packed_idx)
                assert unpack_class_triple_index(packed_idx, num_items) == (first_idx, second_idx, third_idx)

    assert seen == set(range(multiset_triple_count(num_items)))


def test_three_way_class_lookup_subset_matches_direct_counts():
    labels = ["AA", "KK", "QQ", "AKs", "AKo", "KQs", "KQo"]
    artifact = build_three_way_class_lookup(
        class_indices=[class_label_to_id(label) for label in labels],
        processes=1,
        chunk_size=32,
    )

    direct = evaluate_three_way_ranges("AA", "KK", "QQ")
    assert artifact.lookup_labels_ordered("AA", "KK", "QQ") == direct

    direct_mixed = evaluate_three_way_ranges("AKo", "AA", "KQo")
    assert artifact.lookup_labels_ordered("AKo", "AA", "KQo") == direct_mixed

    impossible = artifact.lookup_labels_ordered("AA", "AA", "AA")
    assert impossible.total == 0
    assert impossible.order_counts == (0,) * 13

    legal_count = int(np.count_nonzero(artifact.entries.sum(axis=1)))
    assert artifact.header.legal_count == legal_count
    assert artifact.header.total_runouts == THREE_WAY_CLASS_LOOKUP_TOTAL_RUNOUTS


def test_three_way_class_lookup_permutations_match_direct_queries():
    labels = ["AA", "KK", "QQ"]
    artifact = build_three_way_class_lookup(
        class_indices=[class_label_to_id(label) for label in labels],
        processes=1,
        chunk_size=16,
    )

    for ordered_labels in permutations(labels):
        assert artifact.lookup_labels_ordered(*ordered_labels) == evaluate_three_way_ranges(*ordered_labels)

    # Unordered lookup returns the stored canonical orientation, which for this subset is AA,KK,QQ.
    assert artifact.lookup_labels_unordered("QQ", "AA", "KK") == evaluate_three_way_ranges("AA", "KK", "QQ")


def test_three_way_class_lookup_artifact_roundtrip_and_header(tmp_path):
    labels = ["AA", "KK", "QQ", "AKs"]
    path = tmp_path / "three_way_subset.bin"
    summary = write_three_way_class_lookup(
        path,
        class_indices=[class_label_to_id(label) for label in labels],
        processes=1,
        chunk_size=16,
    )
    artifact = read_three_way_class_lookup(path)
    payload = path.read_bytes()

    assert summary.path == path
    assert artifact.header.universe_class_count == 169
    assert artifact.header.class_count == len(labels)
    assert artifact.header.entry_count == multiset_triple_count(len(labels))
    assert artifact.header.order_count == 13
    assert artifact.header.total_runouts == THREE_WAY_CLASS_LOOKUP_TOTAL_RUNOUTS
    assert payload[:4] == THREE_WAY_CLASS_LOOKUP_MAGIC
    assert int.from_bytes(payload[4:6], "little") == THREE_WAY_CLASS_LOOKUP_VERSION

    assert artifact.lookup_labels_ordered("AA", "KK", "QQ") == evaluate_three_way_ranges("AA", "KK", "QQ")
    assert artifact.lookup_labels_ordered("AA", "AKs", "QQ") == evaluate_three_way_ranges("AA", "AKs", "QQ")
