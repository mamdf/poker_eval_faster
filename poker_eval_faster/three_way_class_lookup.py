from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb
import multiprocessing as mp
import os
from pathlib import Path
import struct
from typing import Iterable, Sequence

import numpy as np

from .main import RANKS_STR, THREE_WAY_ORDER_LABELS, ThreeWayOrderCounts, _range_combo_ids_from_string
from .preflop_canonical import _evaluate_three_way_ranges_preflop_cached


THREE_WAY_CLASS_LOOKUP_MAGIC = b"PTW1"
THREE_WAY_CLASS_LOOKUP_VERSION = 1
THREE_WAY_CLASS_LOOKUP_TOTAL_RUNOUTS = comb(46, 5)

THREE_WAY_CLASS_LOOKUP_HEADER_FORMAT = "<4sHHIIIIIIIII"
THREE_WAY_CLASS_LOOKUP_HEADER_SIZE = struct.calcsize(THREE_WAY_CLASS_LOOKUP_HEADER_FORMAT)
THREE_WAY_CLASS_LOOKUP_CLASS_FORMAT = "<H4sBB"
THREE_WAY_CLASS_LOOKUP_CLASS_SIZE = struct.calcsize(THREE_WAY_CLASS_LOOKUP_CLASS_FORMAT)
THREE_WAY_CLASS_LOOKUP_ENTRY_SIZE = len(THREE_WAY_ORDER_LABELS) * np.dtype(np.uint64).itemsize

_CLASS_META_DTYPE = np.dtype(
    [("class_id", "<u2"), ("label", "S4"), ("combo_count", np.uint8), ("flags", np.uint8)]
)
_ENTRY_DTYPE = np.dtype(np.uint64)
_ORDER_STRENGTHS = (
    (3, 2, 1),
    (3, 1, 2),
    (2, 3, 1),
    (1, 3, 2),
    (2, 1, 3),
    (1, 2, 3),
    (2, 2, 1),
    (2, 1, 2),
    (1, 2, 2),
    (2, 1, 1),
    (1, 2, 1),
    (1, 1, 2),
    (1, 1, 1),
)
_ORDER_STRENGTHS_TO_INDEX = {strengths: idx for idx, strengths in enumerate(_ORDER_STRENGTHS)}

_WORKER_LOCAL_CLASS_COMBO_IDS: tuple[tuple[int, ...], ...] = tuple()


def _generate_hand_classes_169() -> tuple[str, ...]:
    ranks_desc = RANKS_STR[::-1]
    pairs = tuple(rank + rank for rank in ranks_desc)
    suited = tuple(
        high_rank + low_rank + "s"
        for high_idx, high_rank in enumerate(ranks_desc[:-1])
        for low_rank in ranks_desc[high_idx + 1 :]
    )
    offsuit = tuple(
        high_rank + low_rank + "o"
        for high_idx, high_rank in enumerate(ranks_desc[:-1])
        for low_rank in ranks_desc[high_idx + 1 :]
    )
    return pairs + suited + offsuit


HAND_CLASSES_169 = _generate_hand_classes_169()
_HAND_CLASS_LABEL_TO_ID = {label: idx for idx, label in enumerate(HAND_CLASSES_169)}


@dataclass(frozen=True)
class ThreeWayClassLookupHeader:
    universe_class_count: int
    class_count: int
    entry_count: int
    legal_count: int
    order_count: int
    total_runouts: int
    header_size: int
    class_size: int
    entry_size: int


@dataclass(frozen=True)
class ThreeWayClassLookupBuildSummary:
    path: Path
    universe_class_count: int
    class_count: int
    entry_count: int
    legal_count: int
    total_runouts: int


@dataclass(frozen=True)
class ThreeWayClassLookupArtifact:
    header: ThreeWayClassLookupHeader
    classes: np.ndarray
    entries: np.ndarray

    def local_class_index(self, class_id: int) -> int:
        matches = np.where(self.classes["class_id"] == class_id)[0]
        if matches.size == 0:
            raise KeyError(f"Class id {class_id} is not present in this artifact.")
        return int(matches[0])

    def local_class_label(self, local_idx: int) -> str:
        label = self.classes[int(local_idx)]["label"]
        return bytes(label).split(b"\x00", 1)[0].decode("ascii")

    def class_id_label(self, class_id: int) -> str:
        return self.local_class_label(self.local_class_index(class_id))

    def lookup(self, class_a_idx: int, class_b_idx: int, class_c_idx: int) -> ThreeWayOrderCounts:
        return self.lookup_unordered(class_a_idx, class_b_idx, class_c_idx)

    def lookup_unordered(self, class_a_idx: int, class_b_idx: int, class_c_idx: int) -> ThreeWayOrderCounts:
        sorted_indices = tuple(sorted((int(class_a_idx), int(class_b_idx), int(class_c_idx))))
        entry_idx = packed_class_triple_index(
            sorted_indices[0],
            sorted_indices[1],
            sorted_indices[2],
            self.header.class_count,
        )
        counts = tuple(int(value) for value in self.entries[entry_idx].tolist())
        return ThreeWayOrderCounts(order_counts=counts, total=int(sum(counts)))

    def lookup_ordered(self, class_a_idx: int, class_b_idx: int, class_c_idx: int) -> ThreeWayOrderCounts:
        local_indices = (int(class_a_idx), int(class_b_idx), int(class_c_idx))
        sorted_positions = tuple(
            position
            for position, _ in sorted(enumerate(local_indices), key=lambda item: (item[1], item[0]))
        )
        sorted_indices = tuple(local_indices[position] for position in sorted_positions)
        entry_idx = packed_class_triple_index(
            sorted_indices[0],
            sorted_indices[1],
            sorted_indices[2],
            self.header.class_count,
        )
        ordered_counts = _permute_order_counts(
            tuple(int(value) for value in self.entries[entry_idx].tolist()),
            sorted_positions,
        )
        return ThreeWayOrderCounts(order_counts=ordered_counts, total=int(sum(ordered_counts)))

    def lookup_class_ids_unordered(self, class_a_id: int, class_b_id: int, class_c_id: int) -> ThreeWayOrderCounts:
        return self.lookup_unordered(
            self.local_class_index(class_a_id),
            self.local_class_index(class_b_id),
            self.local_class_index(class_c_id),
        )

    def lookup_class_ids_ordered(self, class_a_id: int, class_b_id: int, class_c_id: int) -> ThreeWayOrderCounts:
        return self.lookup_ordered(
            self.local_class_index(class_a_id),
            self.local_class_index(class_b_id),
            self.local_class_index(class_c_id),
        )

    def lookup_labels_unordered(self, class_a_label: str, class_b_label: str, class_c_label: str) -> ThreeWayOrderCounts:
        return self.lookup_class_ids_unordered(
            class_label_to_id(class_a_label),
            class_label_to_id(class_b_label),
            class_label_to_id(class_c_label),
        )

    def lookup_labels_ordered(self, class_a_label: str, class_b_label: str, class_c_label: str) -> ThreeWayOrderCounts:
        return self.lookup_class_ids_ordered(
            class_label_to_id(class_a_label),
            class_label_to_id(class_b_label),
            class_label_to_id(class_c_label),
        )


@lru_cache(maxsize=1)
def all_class_combo_ids() -> tuple[tuple[int, ...], ...]:
    return tuple(_range_combo_ids_from_string(label) for label in HAND_CLASSES_169)


@lru_cache(maxsize=1)
def _three_way_order_permutations() -> dict[tuple[int, int, int], tuple[int, ...]]:
    permutations: dict[tuple[int, int, int], tuple[int, ...]] = {}
    for stored_to_original in (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ):
        remapped_indices: list[int] = []
        for strengths in _ORDER_STRENGTHS:
            remapped = [0, 0, 0]
            for stored_idx, original_idx in enumerate(stored_to_original):
                remapped[original_idx] = strengths[stored_idx]
            remapped_indices.append(_ORDER_STRENGTHS_TO_INDEX[tuple(remapped)])
        permutations[stored_to_original] = tuple(remapped_indices)
    return permutations


def _permute_order_counts(counts: Sequence[int], stored_to_original: tuple[int, int, int]) -> tuple[int, ...]:
    permutation = _three_way_order_permutations()[stored_to_original]
    remapped = [0] * len(THREE_WAY_ORDER_LABELS)
    for src_idx, dest_idx in enumerate(permutation):
        remapped[dest_idx] = int(counts[src_idx])
    return tuple(remapped)


def _normalize_class_label(label: str) -> str:
    normalized = label.strip().upper()
    if len(normalized) == 3 and normalized[2] in ("S", "O"):
        return normalized[:2] + normalized[2].lower()
    return normalized


def class_label_to_id(label: str) -> int:
    normalized = _normalize_class_label(label)
    try:
        return _HAND_CLASS_LABEL_TO_ID[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown hand class label {label!r}") from exc


def class_id_to_label(class_id: int) -> str:
    class_idx = int(class_id)
    if class_idx < 0 or class_idx >= len(HAND_CLASSES_169):
        raise KeyError(f"Unknown hand class id {class_id}")
    return HAND_CLASSES_169[class_idx]


def class_combo_ids(class_ref: int | str) -> tuple[int, ...]:
    if isinstance(class_ref, str):
        class_id = class_label_to_id(class_ref)
    else:
        class_id = int(class_ref)
        if class_id < 0 or class_id >= len(HAND_CLASSES_169):
            raise KeyError(f"Unknown hand class id {class_ref}")
    return all_class_combo_ids()[class_id]


def multiset_triple_count(num_items: int) -> int:
    if num_items < 0:
        raise ValueError(f"num_items must be >= 0, got {num_items}")
    return comb(num_items + 2, 3)


def packed_class_triple_index(first_idx: int, second_idx: int, third_idx: int, num_items: int) -> int:
    first = int(first_idx)
    second = int(second_idx)
    third = int(third_idx)
    total_items = int(num_items)
    if not (0 <= first <= second <= third < total_items):
        raise ValueError(
            f"Packed triple indices must satisfy 0 <= i <= j <= k < {total_items}, "
            f"got {(first_idx, second_idx, third_idx)}"
        )

    prefix_first = comb(total_items + 2, 3) - comb(total_items - first + 2, 3)
    prefix_second = comb(total_items - first + 1, 2) - comb(total_items - second + 1, 2)
    return prefix_first + prefix_second + (third - second)


def unpack_class_triple_index(index: int, num_items: int) -> tuple[int, int, int]:
    packed_idx = int(index)
    total_items = int(num_items)
    entry_count = multiset_triple_count(total_items)
    if packed_idx < 0 or packed_idx >= entry_count:
        raise ValueError(f"Packed triple index must be in [0, {entry_count}), got {packed_idx}")

    remaining = packed_idx
    for first_idx in range(total_items):
        triples_with_first = comb(total_items - first_idx + 1, 2)
        if remaining >= triples_with_first:
            remaining -= triples_with_first
            continue
        for second_idx in range(first_idx, total_items):
            triples_with_second = total_items - second_idx
            if remaining >= triples_with_second:
                remaining -= triples_with_second
                continue
            return first_idx, second_idx, second_idx + remaining

    raise AssertionError("Unable to unpack class triple index")


def _entries_with_first_index(first_idx: int, num_items: int) -> int:
    return comb(num_items - first_idx + 1, 2)


def _iter_chunk_specs(num_items: int, chunk_size: int) -> Iterable[tuple[int, int, int]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    start_idx = 0
    first_start = 0
    while first_start < num_items:
        first_end = first_start
        entry_count = 0
        while first_end < num_items:
            extra = _entries_with_first_index(first_end, num_items)
            if entry_count > 0 and entry_count + extra > chunk_size:
                break
            entry_count += extra
            first_end += 1
            if entry_count >= chunk_size:
                break
        yield start_idx, first_start, first_end
        start_idx += entry_count
        first_start = first_end


def _normalize_class_indices(class_indices: Iterable[int] | None) -> np.ndarray:
    if class_indices is None:
        return np.arange(len(HAND_CLASSES_169), dtype=np.int32)

    selected = np.array(sorted(set(int(idx) for idx in class_indices)), dtype=np.int32)
    if selected.size == 0:
        raise ValueError("class_indices must contain at least one class id.")
    if selected[0] < 0 or selected[-1] >= len(HAND_CLASSES_169):
        raise ValueError(
            f"class_indices must stay within [0, {len(HAND_CLASSES_169) - 1}], got "
            f"{selected[0]}..{selected[-1]}"
        )
    return selected


def _build_class_meta(selected: np.ndarray) -> np.ndarray:
    class_meta = np.zeros(selected.size, dtype=_CLASS_META_DTYPE)
    class_meta["class_id"] = selected.astype(np.uint16)
    all_combo_ids = all_class_combo_ids()
    for local_idx, class_id in enumerate(selected.tolist()):
        label = HAND_CLASSES_169[int(class_id)].encode("ascii")
        class_meta["label"][local_idx] = label
        class_meta["combo_count"][local_idx] = len(all_combo_ids[int(class_id)])
    return class_meta


def _resolve_processes(processes: int | None, entry_count: int, chunk_size: int) -> int:
    if processes is not None:
        if processes < 1:
            raise ValueError(f"processes must be >= 1, got {processes}")
        return processes
    if entry_count <= max(chunk_size * 4, 1024):
        return 1
    return max(os.cpu_count() or 1, 1)


def _init_three_way_class_lookup_worker(local_class_combo_ids: tuple[tuple[int, ...], ...]) -> None:
    global _WORKER_LOCAL_CLASS_COMBO_IDS
    _WORKER_LOCAL_CLASS_COMBO_IDS = local_class_combo_ids


def _evaluate_chunk(spec: tuple[int, int, int]) -> tuple[int, np.ndarray, int]:
    start_idx, first_start, first_end = spec
    rows: list[tuple[int, ...]] = []
    legal_count = 0

    for first_idx in range(first_start, first_end):
        combos_first = _WORKER_LOCAL_CLASS_COMBO_IDS[first_idx]
        for second_idx in range(first_idx, len(_WORKER_LOCAL_CLASS_COMBO_IDS)):
            combos_second = _WORKER_LOCAL_CLASS_COMBO_IDS[second_idx]
            for third_idx in range(second_idx, len(_WORKER_LOCAL_CLASS_COMBO_IDS)):
                combos_third = _WORKER_LOCAL_CLASS_COMBO_IDS[third_idx]
                counts = _evaluate_three_way_ranges_preflop_cached(
                    combos_first,
                    combos_second,
                    combos_third,
                )
                rows.append(counts)
                if any(counts):
                    legal_count += 1

    return start_idx, np.array(rows, dtype=np.uint64), legal_count


def _build_entries(
    local_class_combo_ids: tuple[tuple[int, ...], ...],
    processes: int | None,
    chunk_size: int,
) -> tuple[np.ndarray, int]:
    class_count = len(local_class_combo_ids)
    entry_count = multiset_triple_count(class_count)
    resolved_processes = _resolve_processes(processes, entry_count, chunk_size)
    entries = np.zeros((entry_count, len(THREE_WAY_ORDER_LABELS)), dtype=np.uint64)
    legal_count = 0
    chunk_specs = list(_iter_chunk_specs(class_count, chunk_size))

    if resolved_processes == 1:
        _init_three_way_class_lookup_worker(local_class_combo_ids)
        for spec in chunk_specs:
            start_idx, chunk_rows, chunk_legal = _evaluate_chunk(spec)
            entries[start_idx : start_idx + chunk_rows.shape[0], :] = chunk_rows
            legal_count += chunk_legal
        return entries, legal_count

    with mp.Pool(
        processes=resolved_processes,
        initializer=_init_three_way_class_lookup_worker,
        initargs=(local_class_combo_ids,),
    ) as pool:
        for start_idx, chunk_rows, chunk_legal in pool.imap_unordered(_evaluate_chunk, chunk_specs):
            entries[start_idx : start_idx + chunk_rows.shape[0], :] = chunk_rows
            legal_count += chunk_legal
    return entries, legal_count


def build_three_way_class_lookup(
    class_indices: Iterable[int] | None = None,
    processes: int | None = None,
    chunk_size: int = 4096,
) -> ThreeWayClassLookupArtifact:
    selected = _normalize_class_indices(class_indices)
    class_meta = _build_class_meta(selected)
    local_class_combo_ids = tuple(all_class_combo_ids()[int(class_id)] for class_id in selected.tolist())
    entries, legal_count = _build_entries(local_class_combo_ids, processes=processes, chunk_size=chunk_size)
    header = ThreeWayClassLookupHeader(
        universe_class_count=len(HAND_CLASSES_169),
        class_count=int(selected.size),
        entry_count=int(entries.shape[0]),
        legal_count=legal_count,
        order_count=len(THREE_WAY_ORDER_LABELS),
        total_runouts=THREE_WAY_CLASS_LOOKUP_TOTAL_RUNOUTS,
        header_size=THREE_WAY_CLASS_LOOKUP_HEADER_SIZE,
        class_size=THREE_WAY_CLASS_LOOKUP_CLASS_SIZE,
        entry_size=THREE_WAY_CLASS_LOOKUP_ENTRY_SIZE,
    )
    return ThreeWayClassLookupArtifact(header=header, classes=class_meta, entries=entries)


def write_three_way_class_lookup(
    path: str | Path,
    class_indices: Iterable[int] | None = None,
    processes: int | None = None,
    chunk_size: int = 4096,
) -> ThreeWayClassLookupBuildSummary:
    artifact = build_three_way_class_lookup(
        class_indices=class_indices,
        processes=processes,
        chunk_size=chunk_size,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    header = artifact.header
    header_bytes = struct.pack(
        THREE_WAY_CLASS_LOOKUP_HEADER_FORMAT,
        THREE_WAY_CLASS_LOOKUP_MAGIC,
        THREE_WAY_CLASS_LOOKUP_VERSION,
        0,
        header.universe_class_count,
        header.class_count,
        header.entry_count,
        header.legal_count,
        header.order_count,
        header.total_runouts,
        header.header_size,
        header.class_size,
        header.entry_size,
    )

    with tmp_path.open("wb") as fh:
        fh.write(header_bytes)
        fh.write(artifact.classes.tobytes())
        fh.write(np.asarray(artifact.entries, dtype=_ENTRY_DTYPE).tobytes())
        fh.flush()
        os.fsync(fh.fileno())

    tmp_path.replace(output_path)
    return ThreeWayClassLookupBuildSummary(
        path=output_path,
        universe_class_count=header.universe_class_count,
        class_count=header.class_count,
        entry_count=header.entry_count,
        legal_count=header.legal_count,
        total_runouts=header.total_runouts,
    )


def read_three_way_class_lookup(path: str | Path) -> ThreeWayClassLookupArtifact:
    payload = Path(path).read_bytes()
    header_values = struct.unpack_from(THREE_WAY_CLASS_LOOKUP_HEADER_FORMAT, payload, 0)
    (
        magic,
        version,
        _reserved,
        universe_class_count,
        class_count,
        entry_count,
        legal_count,
        order_count,
        total_runouts,
        header_size,
        class_size,
        entry_size,
    ) = header_values

    if magic != THREE_WAY_CLASS_LOOKUP_MAGIC:
        raise ValueError(f"Unexpected lookup magic {magic!r}")
    if version != THREE_WAY_CLASS_LOOKUP_VERSION:
        raise ValueError(f"Unsupported lookup version {version}")
    if header_size != THREE_WAY_CLASS_LOOKUP_HEADER_SIZE:
        raise ValueError(f"Unexpected header size {header_size}")
    if class_size != THREE_WAY_CLASS_LOOKUP_CLASS_SIZE:
        raise ValueError(f"Unexpected class metadata size {class_size}")
    if order_count != len(THREE_WAY_ORDER_LABELS):
        raise ValueError(f"Unexpected order count {order_count}")
    if entry_size != THREE_WAY_CLASS_LOOKUP_ENTRY_SIZE:
        raise ValueError(f"Unexpected entry size {entry_size}")

    classes_offset = THREE_WAY_CLASS_LOOKUP_HEADER_SIZE
    entries_offset = classes_offset + (class_count * THREE_WAY_CLASS_LOOKUP_CLASS_SIZE)
    expected_size = entries_offset + (entry_count * entry_size)
    if len(payload) != expected_size:
        raise ValueError(f"Unexpected file size {len(payload)} != {expected_size}")

    classes = np.frombuffer(
        payload,
        dtype=_CLASS_META_DTYPE,
        count=class_count,
        offset=classes_offset,
    ).copy()
    flat_entries = np.frombuffer(
        payload,
        dtype=_ENTRY_DTYPE,
        count=entry_count * len(THREE_WAY_ORDER_LABELS),
        offset=entries_offset,
    ).copy()
    entries = flat_entries.reshape(entry_count, len(THREE_WAY_ORDER_LABELS))
    header = ThreeWayClassLookupHeader(
        universe_class_count=universe_class_count,
        class_count=class_count,
        entry_count=entry_count,
        legal_count=legal_count,
        order_count=order_count,
        total_runouts=total_runouts,
        header_size=header_size,
        class_size=class_size,
        entry_size=entry_size,
    )
    return ThreeWayClassLookupArtifact(header=header, classes=classes, entries=entries)
