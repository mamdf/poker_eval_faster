"""Build or check the preflop equity tables shipped in poker_eval_faster/data.

    build [--only hu|3way|random ...]   hu ~2 min, 3way many hours (resumable), random ~10 min
    build --only 3way --legacy-3way PATH convert a verified legacy 3-way pickle in seconds
    check --old-hu PATH --old-3way PATH  compare with legacy pickled class tables
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from math import comb, log, sqrt
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_eval_faster import (  # noqa: E402
    HAND_CLASSES_169,
    class_combo_ids,
    combo_id_to_cards,
    estimate_equity_vs_random,
    evaluate_ranges,
    evaluate_three_way_ranges,
    multiset_triple_count,
    packed_class_triple_index,
)
from poker_eval_faster.eval_cython.three_way_class_lookup_builder import (  # noqa: E402
    evaluate_three_way_class_counts_c,
)
from poker_eval_faster.main import RANKS_STR  # noqa: E402
from poker_eval_faster.preflop_tables import (  # noqa: E402
    PREFLOP_EXACT_OPPONENTS,
    PREFLOP_HU_FILE,
    PREFLOP_MAX_OPPONENTS,
    PREFLOP_RANDOM_SAMPLES,
    PREFLOP_THREE_WAY_FILE,
    PREFLOP_VS_RANDOM_FILE,
    _expected_vs_random_runouts,
    _heads_up_class_counts,
    _legal_triple_counts,
    _three_way_equity_rows,
    _vs_two_random,
    preflop_class_equities,
    preflop_equity_vs_random,
    preflop_random_seed,
)

NUM_CLASSES = len(HAND_CLASSES_169)
STEPS = ("hu", "3way", "random")
_CLASS_COMBOS: tuple[np.ndarray, ...] = ()


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _load_work(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"{path} is missing: run the step that builds it first")
    return np.load(path, allow_pickle=False)


def build_hu(work_dir: Path, data_dir: Path) -> None:
    started = time.perf_counter()
    num2, total = _heads_up_class_counts(range(NUM_CLASSES))
    assert np.array_equal(np.diag(num2), np.diag(total)), "a class against itself must be exactly 0.5"
    _save(work_dir / "hu_num2.npy", num2)
    _save(work_dir / "hu_total.npy", total)
    _save(data_dir / PREFLOP_HU_FILE, num2 / (2.0 * total))
    print(f"hu: {NUM_CLASSES}x{NUM_CLASSES} classes in {time.perf_counter() - started:.0f} s")


def _init_worker() -> None:
    global _CLASS_COMBOS
    _CLASS_COMBOS = tuple(np.array(class_combo_ids(class_id), dtype=np.int32) for class_id in range(NUM_CLASSES))


def _three_way_block(first: int) -> tuple[int, np.ndarray]:
    rows = [
        evaluate_three_way_class_counts_c(_CLASS_COMBOS[first], _CLASS_COMBOS[second], _CLASS_COMBOS[third])
        for second in range(first, NUM_CLASSES)
        for third in range(second, NUM_CLASSES)
    ]
    return first, np.array(rows, dtype=np.uint64)


def _save_three_way(work_dir: Path, data_dir: Path, equities: np.ndarray, deals: np.ndarray) -> None:
    assert np.array_equal(np.isnan(equities[:, 0]), deals == 0), "impossible triples must have no legal deal"
    _save(work_dir / "three_way_equity.npy", equities)
    _save(work_dir / "three_way_deals.npy", deals)
    _save(data_dir / PREFLOP_THREE_WAY_FILE, equities.astype(np.float32))
    print(f"3way: wrote {len(equities)} triples")


def convert_legacy_three_way(work_dir: Path, data_dir: Path, path: Path) -> None:
    labels = _legacy_labels()
    with path.open("rb") as handle:
        legacy = pickle.load(handle)  # Trusted local file only.
    equities = np.full((multiset_triple_count(NUM_CLASSES), 3), np.nan)
    for key, values in legacy.items():
        seats = sorted(zip((HAND_CLASSES_169.index(labels[i]) for i in key), values))
        equities[packed_class_triple_index(*(class_id for class_id, _ in seats), NUM_CLASSES)] = [v for _, v in seats]
    assert len(legacy) == len(equities)
    _save_three_way(work_dir, data_dir, equities, _legal_triple_counts(range(NUM_CLASSES)))


def build_three_way(work_dir: Path, data_dir: Path, processes: int, force: bool) -> None:
    block_dir = work_dir / "three_way"
    pending = [first for first in range(NUM_CLASSES) if force or not (block_dir / f"first_{first:03d}.npy").exists()]
    started = time.perf_counter()
    if pending:
        print(f"3way: {len(pending)} blocks pending on {processes} processes")
        with mp.Pool(processes, initializer=_init_worker) as pool:
            for done, (first, rows) in enumerate(pool.imap_unordered(_three_way_block, pending), start=1):
                _save(block_dir / f"first_{first:03d}.npy", rows)
                print(f"3way: block {first} ({len(rows)} rows), {done}/{len(pending)}, "
                      f"{time.perf_counter() - started:.0f} s", flush=True)

    blocks = []
    for first in range(NUM_CLASSES):
        block = _load_work(block_dir / f"first_{first:03d}.npy")
        assert sum(len(b) for b in blocks) == packed_class_triple_index(first, first, first, NUM_CLASSES)
        blocks.append(block)
    entries = np.concatenate(blocks)
    assert entries.shape == (multiset_triple_count(NUM_CLASSES), 13)
    runouts = entries.astype(np.int64).sum(axis=1)
    assert np.all(runouts % comb(46, 5) == 0)
    _save_three_way(work_dir, data_dir, _three_way_equity_rows(entries), runouts // comb(46, 5))


def _hoeffding_radius(samples: int, alpha: float = 1e-9) -> float:
    return sqrt(log(2.0 / alpha) / (2.0 * samples))


def _estimate(class_id: int, num_opponents: int, samples: int) -> float:
    hero = combo_id_to_cards(class_combo_ids(class_id)[0])
    seed = preflop_random_seed(num_opponents, class_id)
    return estimate_equity_vs_random(hero, [], num_opponents, samples=samples, seed=seed).equity


def build_random(work_dir: Path, data_dir: Path, processes: int, samples: int) -> None:
    num2 = _load_work(work_dir / "hu_num2.npy")
    total2 = _load_work(work_dir / "hu_total.npy")
    equities = np.nan_to_num(_load_work(work_dir / "three_way_equity.npy"))
    deals = _load_work(work_dir / "three_way_deals.npy")
    pot3, deals3 = _vs_two_random(equities * deals[:, None], deals, NUM_CLASSES)
    table = np.zeros((NUM_CLASSES, PREFLOP_MAX_OPPONENTS), dtype=np.float64)
    for class_id in range(NUM_CLASSES):
        assert total2[class_id].sum() == _expected_vs_random_runouts(class_id, 1)
        assert deals3[class_id] * comb(46, 5) == _expected_vs_random_runouts(class_id, 2)
        table[class_id, 0] = num2[class_id].sum() / (2.0 * total2[class_id].sum())
        table[class_id, 1] = pot3[class_id] / deals3[class_id]

    # Monte Carlo for 3+ opponents, plus a cross-check of the exact columns.
    tasks = [(class_id, n) for n in range(1, PREFLOP_MAX_OPPONENTS + 1) for class_id in range(NUM_CLASSES)]
    started = time.perf_counter()
    with ThreadPoolExecutor(processes) as pool:
        futures = {task: pool.submit(_estimate, *task, samples) for task in tasks}
        for done, (task, future) in enumerate(futures.items(), start=1):
            estimate = future.result()
            class_id, n = task
            if n <= PREFLOP_EXACT_OPPONENTS:
                error = abs(estimate - table[class_id, n - 1])
                assert error <= _hoeffding_radius(samples), (HAND_CLASSES_169[class_id], n, error)
            else:
                table[class_id, n - 1] = estimate
            if done % NUM_CLASSES == 0:
                print(f"random: {done}/{len(tasks)} estimates, {time.perf_counter() - started:.0f} s", flush=True)
    _save(data_dir / PREFLOP_VS_RANDOM_FILE, table)
    print(f"random: wrote {table.shape} table")


def _legacy_labels() -> list[str]:
    labels = [_legacy_label(index) for index in range(NUM_CLASSES)]
    assert sorted(labels) == sorted(HAND_CLASSES_169)
    return labels


def _legacy_label(index: int) -> str:
    """Class label of the legacy pickles: pairs/offsuit by (high, low) rank, then suited."""
    suited = index >= 91
    offset = index - 91 if suited else index

    def first_offset(high: int) -> int:
        return high * (high - 1) // 2 if suited else high * (high + 1) // 2

    high = 0
    while first_offset(high + 1) <= offset:
        high += 1
    low = offset - first_offset(high)
    label = RANKS_STR[high] + RANKS_STR[low]
    return label if high == low else label + ("s" if suited else "o")


def check(data_dir: Path, old_hu: Path | None, old_three_way: Path | None, sample: int) -> bool:
    if data_dir.resolve() != (ROOT / "poker_eval_faster" / "data").resolve():
        raise SystemExit("check reads the package tables; run it against poker_eval_faster/data")
    failures: list[str] = []

    hu = np.load(data_dir / PREFLOP_HU_FILE)
    three_way = np.load(data_dir / PREFLOP_THREE_WAY_FILE)
    vs_random = np.load(data_dir / PREFLOP_VS_RANDOM_FILE)
    if hu.shape != (NUM_CLASSES, NUM_CLASSES) or not np.allclose(hu + hu.T, 1.0, rtol=0, atol=1e-12):
        failures.append("HU table is not antisymmetric")
    legal = ~np.isnan(three_way[:, 0])
    if three_way.shape != (multiset_triple_count(NUM_CLASSES), 3) or (~legal).sum() != 325:
        failures.append(f"3-way table has {(~legal).sum()} impossible rows, expected 325")
    if not np.allclose(three_way[legal].sum(axis=1), 1.0, rtol=0, atol=1e-5):
        failures.append("3-way rows do not sum to 1")
    if vs_random.shape != (NUM_CLASSES, PREFLOP_MAX_OPPONENTS) or not np.all(np.diff(vs_random, axis=1) < 0):
        failures.append("equity vs random is not decreasing in the number of opponents")

    labels = _legacy_labels()
    if old_hu is not None:
        with old_hu.open("rb") as handle:
            legacy = pickle.load(handle)  # Trusted local file only.
        error = max(
            max(abs(a - b) for a, b in zip(preflop_class_equities((labels[i], labels[j])), values))
            for (i, j), values in legacy.items()
        )
        print(f"check: legacy HU {len(legacy)} entries, max error {error:.2e}")
        if len(legacy) != 14365 or error > 1e-12:
            failures.append("HU table differs from the legacy file")
    if old_three_way is not None:
        with old_three_way.open("rb") as handle:
            legacy = pickle.load(handle)  # Trusted local file only.
        error, nan_mismatch = 0.0, 0
        for key, values in legacy.items():
            classes = tuple(labels[i] for i in key)
            try:
                equities = preflop_class_equities(classes)
            except ValueError:
                nan_mismatch += not np.isnan(values[0])
                continue
            nan_mismatch += bool(np.isnan(values[0]))
            error = max(error, *(abs(a - b) for a, b in zip(equities, values)))
        print(f"check: legacy 3-way {len(legacy)} entries, max error {error:.2e}, NaN mismatches {nan_mismatch}")
        if len(legacy) != 818805 or error > 2e-6 or nan_mismatch:
            failures.append("3-way table differs from the legacy file")

    rng = random.Random(0)
    hu_error = three_way_error = 0.0
    for _ in range(sample):
        a, b = rng.sample(HAND_CLASSES_169, 2)
        hu_error = max(hu_error, abs(preflop_class_equities((a, b))[0] - evaluate_ranges(a, b)))
        classes = rng.sample(HAND_CLASSES_169, 3)
        try:
            equities = preflop_class_equities(classes)
        except ValueError:
            continue
        expected = evaluate_three_way_ranges(*classes).equities
        three_way_error = max(three_way_error, *(abs(x - y) for x, y in zip(equities, expected)))
    if sample:
        print(f"check: {sample} recomputed samples, max error HU {hu_error:.2e}, 3-way {three_way_error:.2e}")
    if hu_error > 1e-12 or three_way_error > 2e-6:
        failures.append("tables differ from recomputed library values")

    for hand, expected in (("AA", 0.8520371330210104), ("72o", 0.34583647315344157),
                           ("JTs", 0.5752785710757826), ("65o", 0.39944302303939544)):
        if abs(preflop_equity_vs_random(hand) - expected) > 1e-12:
            failures.append(f"{hand} vs one random differs from the exact reference")

    for failure in failures:
        print(f"FAIL: {failure}")
    print("check: OK" if not failures else f"check: {len(failures)} failures")
    return not failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--only", action="append", choices=STEPS, help="Build only this step (repeatable).")
    parser.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--samples", type=int, default=PREFLOP_RANDOM_SAMPLES)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build" / "preflop_tables")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "poker_eval_faster" / "data")
    parser.add_argument("--force", action="store_true", help="Recompute existing 3-way blocks.")
    parser.add_argument("--legacy-3way", type=Path, help="Build the 3-way step from this legacy pickle instead.")
    parser.add_argument("--old-hu", type=Path, help="Legacy pickle with (i, j) -> [eq_i, eq_j].")
    parser.add_argument("--old-3way", type=Path, help="Legacy pickle with (i, j, k) -> [eq_i, eq_j, eq_k].")
    parser.add_argument("--sample", type=int, default=50, help="Entries to recompute with the library.")
    args = parser.parse_args()

    if args.command == "check":
        sys.exit(0 if check(args.data_dir, args.old_hu, args.old_3way, args.sample) else 1)
    steps = args.only or STEPS
    if "hu" in steps:
        build_hu(args.work_dir, args.data_dir)
    if "3way" in steps and args.legacy_3way:
        convert_legacy_three_way(args.work_dir, args.data_dir, args.legacy_3way)
    elif "3way" in steps:
        build_three_way(args.work_dir, args.data_dir, args.processes, args.force)
    if "random" in steps:
        build_random(args.work_dir, args.data_dir, args.processes, args.samples)


if __name__ == "__main__":
    main()
