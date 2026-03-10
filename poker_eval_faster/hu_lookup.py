from __future__ import annotations

from dataclasses import dataclass
from math import comb
import os
from pathlib import Path
import struct
from typing import Iterable

import numpy as np

from .eval_cython.hands_evaluate import evaluate_heads_up_counts_c
from .main import HeadsUpCounts, canonical_combo_masks, canonical_combos, int_to_cards, packed_pair_index


HU_LOOKUP_MAGIC = b"PHU1"
HU_LOOKUP_VERSION = 1
HU_LOOKUP_SENTINEL = np.uint32(0xFFFFFFFF)
HU_LOOKUP_TOTAL_RUNOUTS = comb(48, 5)

HU_LOOKUP_HEADER_FORMAT = "<4sHHIIIIIIII"
HU_LOOKUP_HEADER_SIZE = struct.calcsize(HU_LOOKUP_HEADER_FORMAT)
HU_LOOKUP_COMBO_FORMAT = "<HBBQ"
HU_LOOKUP_COMBO_SIZE = struct.calcsize(HU_LOOKUP_COMBO_FORMAT)
HU_LOOKUP_ENTRY_FORMAT = "<II"
HU_LOOKUP_ENTRY_SIZE = struct.calcsize(HU_LOOKUP_ENTRY_FORMAT)

_EMPTY_BOARD = np.array([], dtype="int32")
_COMBO_META_DTYPE = np.dtype(
    [("combo_id", "<u2"), ("card_a", np.uint8), ("card_b", np.uint8), ("mask", "<u8")]
)
_ENTRY_DTYPE = np.dtype([("win", "<u4"), ("tie", "<u4")])


@dataclass(frozen=True)
class HuLookupHeader:
    combo_count: int
    entry_count: int
    legal_count: int
    total_runouts: int
    sentinel: int
    header_size: int
    combo_size: int
    entry_size: int


@dataclass(frozen=True)
class HuLookupBuildSummary:
    path: Path
    combo_count: int
    entry_count: int
    legal_count: int
    total_runouts: int


@dataclass(frozen=True)
class HuLookupArtifact:
    header: HuLookupHeader
    combos: np.ndarray
    entries: np.ndarray

    def local_combo_index(self, combo_id: int) -> int:
        matches = np.where(self.combos["combo_id"] == combo_id)[0]
        if matches.size == 0:
            raise KeyError(f"Combo id {combo_id} is not present in this artifact.")
        return int(matches[0])

    def local_combo_cards(self, local_idx: int) -> list[str]:
        row = self.combos[int(local_idx)]
        return int_to_cards([int(row["card_a"]), int(row["card_b"])])

    def local_combo_str(self, local_idx: int) -> str:
        return "".join(self.local_combo_cards(local_idx))

    def combo_id_cards(self, combo_id: int) -> list[str]:
        return self.local_combo_cards(self.local_combo_index(combo_id))

    def combo_id_str(self, combo_id: int) -> str:
        return self.local_combo_str(self.local_combo_index(combo_id))

    def lookup(self, combo_a_idx: int, combo_b_idx: int) -> HeadsUpCounts | None:
        return self.lookup_unordered(combo_a_idx, combo_b_idx)

    def lookup_unordered(self, combo_a_idx: int, combo_b_idx: int) -> HeadsUpCounts | None:
        if combo_a_idx == combo_b_idx:
            return None
        idx = packed_pair_index(combo_a_idx, combo_b_idx, self.header.combo_count)
        win = int(self.entries["win"][idx])
        tie = int(self.entries["tie"][idx])
        if win == self.header.sentinel and tie == self.header.sentinel:
            return None
        return HeadsUpCounts(wins=win, ties=tie, total=self.header.total_runouts)

    def lookup_ordered(self, combo_a_idx: int, combo_b_idx: int) -> HeadsUpCounts | None:
        counts = self.lookup_unordered(combo_a_idx, combo_b_idx)
        if counts is None:
            return None
        if combo_a_idx < combo_b_idx:
            return counts
        return HeadsUpCounts(wins=counts.losses, ties=counts.ties, total=counts.total)

    def lookup_combo_ids_unordered(self, combo_a_id: int, combo_b_id: int) -> HeadsUpCounts | None:
        return self.lookup_unordered(self.local_combo_index(combo_a_id), self.local_combo_index(combo_b_id))

    def lookup_combo_ids_ordered(self, combo_a_id: int, combo_b_id: int) -> HeadsUpCounts | None:
        return self.lookup_ordered(self.local_combo_index(combo_a_id), self.local_combo_index(combo_b_id))


def triangular_pair_count(combo_count: int) -> int:
    return (combo_count * (combo_count - 1)) // 2


def legal_pair_count(combo_masks: np.ndarray) -> int:
    count = 0
    combo_count = combo_masks.shape[0]
    for first_idx in range(combo_count):
        first_mask = int(combo_masks[first_idx])
        for second_idx in range(first_idx + 1, combo_count):
            if first_mask & int(combo_masks[second_idx]):
                continue
            count += 1
    return count


def build_hu_preflop_lookup(combo_indices: Iterable[int] | None = None) -> HuLookupArtifact:
    all_combos = canonical_combos()
    all_masks = canonical_combo_masks()

    if combo_indices is None:
        selected = np.arange(all_combos.shape[0], dtype=np.int32)
    else:
        selected = np.array(sorted(set(int(idx) for idx in combo_indices)), dtype=np.int32)

    combos_int32 = all_combos[selected]
    combo_count = int(combos_int32.shape[0])
    combo_meta = np.zeros(combo_count, dtype=_COMBO_META_DTYPE)
    combo_meta["combo_id"] = selected.astype(np.uint16)
    combo_meta["card_a"] = combos_int32[:, 0].astype(np.uint8)
    combo_meta["card_b"] = combos_int32[:, 1].astype(np.uint8)
    combo_meta["mask"] = all_masks[selected]

    entry_count = triangular_pair_count(combo_count)
    entries = np.zeros(entry_count, dtype=_ENTRY_DTYPE)
    entries["win"].fill(HU_LOOKUP_SENTINEL)
    entries["tie"].fill(HU_LOOKUP_SENTINEL)

    legal_count = 0
    for first_idx in range(combo_count):
        first_mask = int(combo_meta["mask"][first_idx])
        for second_idx in range(first_idx + 1, combo_count):
            pair_idx = packed_pair_index(first_idx, second_idx, combo_count)
            if first_mask & int(combo_meta["mask"][second_idx]):
                continue

            counts = evaluate_heads_up_counts_c(combos_int32[first_idx], combos_int32[second_idx], _EMPTY_BOARD)
            entries["win"][pair_idx] = int(counts[0])
            entries["tie"][pair_idx] = int(counts[1])
            legal_count += 1

    header = HuLookupHeader(
        combo_count=combo_count,
        entry_count=entry_count,
        legal_count=legal_count,
        total_runouts=HU_LOOKUP_TOTAL_RUNOUTS,
        sentinel=int(HU_LOOKUP_SENTINEL),
        header_size=HU_LOOKUP_HEADER_SIZE,
        combo_size=HU_LOOKUP_COMBO_SIZE,
        entry_size=HU_LOOKUP_ENTRY_SIZE,
    )
    return HuLookupArtifact(header=header, combos=combo_meta, entries=entries)


def write_hu_preflop_lookup(path: str | Path, combo_indices: Iterable[int] | None = None) -> HuLookupBuildSummary:
    artifact = build_hu_preflop_lookup(combo_indices=combo_indices)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    header = artifact.header
    header_bytes = struct.pack(
        HU_LOOKUP_HEADER_FORMAT,
        HU_LOOKUP_MAGIC,
        HU_LOOKUP_VERSION,
        0,
        header.combo_count,
        header.entry_count,
        header.legal_count,
        header.total_runouts,
        header.sentinel,
        header.header_size,
        header.combo_size,
        header.entry_size,
    )

    with tmp_path.open("wb") as fh:
        fh.write(header_bytes)
        fh.write(artifact.combos.tobytes())
        fh.write(artifact.entries.tobytes())
        fh.flush()
        os.fsync(fh.fileno())

    tmp_path.replace(output_path)
    return HuLookupBuildSummary(
        path=output_path,
        combo_count=header.combo_count,
        entry_count=header.entry_count,
        legal_count=header.legal_count,
        total_runouts=header.total_runouts,
    )


def read_hu_preflop_lookup(path: str | Path) -> HuLookupArtifact:
    payload = Path(path).read_bytes()
    header_values = struct.unpack_from(HU_LOOKUP_HEADER_FORMAT, payload, 0)
    (
        magic,
        version,
        _reserved,
        combo_count,
        entry_count,
        legal_count,
        total_runouts,
        sentinel,
        header_size,
        combo_size,
        entry_size,
    ) = header_values

    if magic != HU_LOOKUP_MAGIC:
        raise ValueError(f"Unexpected lookup magic {magic!r}")
    if version != HU_LOOKUP_VERSION:
        raise ValueError(f"Unsupported lookup version {version}")
    if header_size != HU_LOOKUP_HEADER_SIZE:
        raise ValueError(f"Unexpected header size {header_size}")
    if combo_size != HU_LOOKUP_COMBO_SIZE:
        raise ValueError(f"Unexpected combo record size {combo_size}")
    if entry_size != HU_LOOKUP_ENTRY_SIZE:
        raise ValueError(f"Unexpected entry record size {entry_size}")

    combos_offset = HU_LOOKUP_HEADER_SIZE
    entries_offset = combos_offset + (combo_count * HU_LOOKUP_COMBO_SIZE)
    expected_size = entries_offset + (entry_count * HU_LOOKUP_ENTRY_SIZE)
    if len(payload) != expected_size:
        raise ValueError(f"Unexpected file size {len(payload)} != {expected_size}")

    combos = np.frombuffer(payload, dtype=_COMBO_META_DTYPE, count=combo_count, offset=combos_offset).copy()
    entries = np.frombuffer(payload, dtype=_ENTRY_DTYPE, count=entry_count, offset=entries_offset).copy()
    header = HuLookupHeader(
        combo_count=combo_count,
        entry_count=entry_count,
        legal_count=legal_count,
        total_runouts=total_runouts,
        sentinel=sentinel,
        header_size=header_size,
        combo_size=combo_size,
        entry_size=entry_size,
    )
    return HuLookupArtifact(header=header, combos=combos, entries=entries)
