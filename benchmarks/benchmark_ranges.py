#!/usr/bin/env python3
import argparse
import time

from poker_eval_faster import evaluate_hands, evaluate_heads_up_counts, evaluate_ranges
from poker_eval_faster.main import (
    _clear_preflop_caches,
    _preflop_range_matchup_profile,
    _range_combo_ids_from_string,
)


CASES = [
    ("simple", "AA", "KK"),
    ("complex", "99+,AQs+,AQo+", "JJ-77,AQs-A9s,KJs+,QJs,AQo-AJo,KQo"),
    (
        "bounded",
        "A2s,K2s,Q2s,T6s,95s,84s,62s,52s",
        "Q5o-Q2o,J9o-J7o,T7o-T6o,96o-95o,85o-84o,74o,63o,53o",
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


def _benchmark_baselines(iterations: int) -> None:
    heads_up, heads_up_elapsed = _time_once(
        lambda: _repeat(iterations, lambda: evaluate_heads_up_counts(["As", "Ah"], ["Ks", "Kh"], []))
    )
    hands, hands_elapsed = _time_once(
        lambda: _repeat(iterations, lambda: evaluate_hands([["As", "Ah"], ["Ks", "Kh"]]))
    )
    print("Baselines")
    print(f"  evaluate_heads_up_counts: {(heads_up_elapsed / iterations) * 1e6:.2f} us/op")
    print(f"  evaluate_hands: {(hands_elapsed / iterations) * 1e6:.2f} us/op")
    print(f"  sample counts equity: {heads_up.equity:.6f}")
    print(f"  sample hands equity: {hands[0]:.6f}\n")


def _benchmark_case(name: str, hero: str, villain: str, warm_iterations: int) -> None:
    _clear_preflop_caches()
    equity, cold_elapsed = _time_once(lambda: evaluate_ranges(hero, villain, []))

    hero_ids = _range_combo_ids_from_string(hero)
    villain_ids = _range_combo_ids_from_string(villain)
    profile = _preflop_range_matchup_profile(hero_ids, villain_ids)
    legal_pairs = sum(multiplicity for _, multiplicity in profile)
    canonical_matchups = len(profile)

    _, warm_elapsed = _time_once(lambda: _repeat(warm_iterations, lambda: evaluate_ranges(hero, villain, [])))
    warm_per_op = warm_elapsed / warm_iterations

    print(f"{name}: {hero} vs {villain}")
    print(
        f"  combos={len(hero_ids)}x{len(villain_ids)} "
        f"legal_pairs={legal_pairs} canonical_matchups={canonical_matchups}"
    )
    print(
        f"  reduction={legal_pairs / canonical_matchups:.2f}x "
        f"equity={equity:.6f}"
    )
    print(
        f"  cold={cold_elapsed * 1e3:.2f} ms "
        f"warm={warm_per_op * 1e6:.2f} us/op\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark preflop range-vs-range exact evaluation")
    parser.add_argument(
        "--warm-iters",
        type=int,
        default=200,
        help="Iterations for the warm benchmark after caches are populated",
    )
    parser.add_argument(
        "--baseline-iters",
        type=int,
        default=100,
        help="Iterations for combo-vs-combo reference timings",
    )
    args = parser.parse_args()

    _benchmark_baselines(args.baseline_iters)
    for name, hero, villain in CASES:
        _benchmark_case(name, hero, villain, args.warm_iters)


if __name__ == "__main__":
    main()
