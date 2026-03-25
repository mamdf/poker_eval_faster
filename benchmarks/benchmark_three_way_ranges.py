#!/usr/bin/env python3
import argparse
import time

from poker_eval_faster import evaluate_three_way_orders, evaluate_three_way_ranges
from poker_eval_faster.main import (
    _clear_preflop_caches,
    _preflop_three_way_range_matchup_profile,
    _range_combo_ids_from_string,
)


CASES = [
    ("simple", "AA", "KK", "QQ"),
    (
        "bounded",
        "AKs,AKo,QQ,JJ",
        "TT-99,AQs-AJs,KQs",
        "88-77,ATs-A8s,KJs+,QJs,AJo",
    ),
]


def _time_once(func):
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    return result, elapsed


def _repeat(iterations: int, func):
    last_result = None
    for _ in range(iterations):
        last_result = func()
    return last_result


def _benchmark_baseline(iterations: int) -> None:
    sample_hands = [["As", "Ah"], ["Ks", "Kh"], ["Qs", "Qh"]]
    sample_result, elapsed = _time_once(
        lambda: _repeat(iterations, lambda: evaluate_three_way_orders(sample_hands))
    )
    print("Baseline")
    print(f"  evaluate_three_way_orders: {(elapsed / iterations) * 1e6:.2f} us/op")
    print(f"  sample equities: {sample_result.equities}\n")


def _benchmark_case(name: str, range_a: str, range_b: str, range_c: str, warm_iterations: int) -> None:
    _clear_preflop_caches()
    result, cold_elapsed = _time_once(lambda: evaluate_three_way_ranges(range_a, range_b, range_c))

    range_a_ids = _range_combo_ids_from_string(range_a)
    range_b_ids = _range_combo_ids_from_string(range_b)
    range_c_ids = _range_combo_ids_from_string(range_c)
    profile = _preflop_three_way_range_matchup_profile(range_a_ids, range_b_ids, range_c_ids)
    legal_triples = sum(multiplicity for _, multiplicity in profile)
    canonical_matchups = len(profile)

    _, warm_elapsed = _time_once(
        lambda: _repeat(warm_iterations, lambda: evaluate_three_way_ranges(range_a, range_b, range_c))
    )

    print(f"{name}: {range_a} vs {range_b} vs {range_c}")
    print(
        f"  combos={len(range_a_ids)}x{len(range_b_ids)}x{len(range_c_ids)} "
        f"legal_triples={legal_triples} canonical_matchups={canonical_matchups}"
    )
    print(
        f"  reduction={legal_triples / canonical_matchups:.2f}x "
        f"equities={tuple(round(value, 6) for value in result.equities)}"
    )
    print(
        f"  cold={cold_elapsed * 1e3:.2f} ms "
        f"warm={(warm_elapsed / warm_iterations) * 1e6:.2f} us/op\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exact preflop three-range evaluation")
    parser.add_argument(
        "--warm-iters",
        type=int,
        default=50,
        help="Iterations for the warm benchmark after caches are populated",
    )
    parser.add_argument(
        "--baseline-iters",
        type=int,
        default=20,
        help="Iterations for the fixed three-hand preflop reference benchmark",
    )
    args = parser.parse_args()

    _benchmark_baseline(args.baseline_iters)
    for case in CASES:
        _benchmark_case(*case, warm_iterations=args.warm_iters)


if __name__ == "__main__":
    main()
