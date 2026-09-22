"""Bounded exact 3-max postflop benchmark: python benchmarks/benchmark_two_random.py."""
import json
from statistics import median
from time import perf_counter

from poker_eval_faster import evaluate_one_hand_vs_two_random


def main():
    # Inputs adapted from holdem_insights/examples/holdem_indicator.
    for hero, board in [
        (["Ac", "Tc"], ["9c", "2d", "As", "2h", "7h"]),
        (["5h", "6d"], ["5d", "7h", "Qc", "Qs", "3h"]),
        (["4h", "7d"], ["9d", "8c", "9c", "2h", "As"]),
        (["Ks", "Jh"], ["4c", "5d", "5h", "9s", "2c"]),
    ]:
        for size in (3, 4, 5):
            times = []
            for _ in range(21):
                start = perf_counter()
                counts = evaluate_one_hand_vs_two_random(hero, board[:size])
                times.append(1000 * (perf_counter() - start))
            print(json.dumps({
                "hero": hero, "board": board[:size], "counts": counts,
                "total": sum(counts), "first_ms": times[0],
                "warm_median_ms": median(times[1:]), "warm_max_ms": max(times[1:]),
            }))


if __name__ == "__main__":
    main()
