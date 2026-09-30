"""Random-opponent Monte Carlo latency on every street, including Python overhead."""
import argparse
import json
import platform
from statistics import median
from time import perf_counter


import_start = perf_counter()
from poker_eval_faster import estimate_equity_vs_random
import_ms = 1000 * (perf_counter() - import_start)


SCENARIOS = [
    (["Ac", "Tc"], ["9c", "2d", "As", "2h", "7h"]),
    (["5h", "6d"], ["5d", "7h", "Qc", "Qs", "3h"]),
    (["4h", "7d"], ["9d", "8c", "9c", "2h", "As"]),
    (["Ks", "Jh"], ["4c", "5d", "5h", "9s", "2c"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", nargs="+", type=int, default=[2, 3, 6, 8, 10], choices=range(2, 11))
    parser.add_argument("--samples", nargs="+", type=int, default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    if args.repeats < 2 or any(n < 1 for n in args.samples):
        parser.error("Use at least two repeats and positive sample counts")
    print(json.dumps({"package_import_ms": import_ms, "python": platform.python_version(),
                      "platform": platform.platform(), "repeats": args.repeats}))
    for hero, board in SCENARIOS:
        for size in (0, 3, 4, 5):
            for players in args.players:
                for samples in args.samples:
                    times = []
                    for repetition in range(args.repeats + 1):
                        start = perf_counter()
                        result = estimate_equity_vs_random(
                            hero, board[:size], players - 1, samples=samples, seed=42 + repetition,
                        )
                        times.append(1000 * (perf_counter() - start))
                    warm = sorted(times[1:])
                    # Linear interpolation, matching numpy.percentile's default.
                    position = 0.95 * (len(warm) - 1)
                    lower = int(position)
                    p95 = warm[lower] + (position - lower) * (warm[min(lower + 1, len(warm) - 1)] - warm[lower])
                    row = dict(hero=hero, board=board[:size], players=players, samples=samples,
                               equity=result.equity, error_bound_95=result.error_bound_95,
                               first_ms=times[0], warm_median_ms=median(warm), warm_p95_ms=p95,
                               samples_per_second=samples * 1000 / median(warm))
                    if players == 8 and samples == 100_000:
                        row["meets_100ms_target"] = p95 <= 100
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
