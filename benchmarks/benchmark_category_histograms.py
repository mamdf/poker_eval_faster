"""Run with python benchmarks/benchmark_category_histograms.py."""
import json
from statistics import median
from time import perf_counter

from poker_eval_faster import river_category_histograms


def main():
    hero = ["Ac", "Tc"]
    for size in (3, 4, 5):
        board = ["9c", "2d", "As", "2h", "7h"][:size]
        times = []
        for _ in range(101):
            start = perf_counter()
            result = river_category_histograms(hero, board)
            times.append(1000 * (perf_counter() - start))
        print(json.dumps({
            "board": board, "hero_total": result["hero_total"],
            "opponent_total": result["opponent_total"],
            "first_ms": times[0], "warm_median_ms": median(times[1:]),
        }))


if __name__ == "__main__":
    main()
