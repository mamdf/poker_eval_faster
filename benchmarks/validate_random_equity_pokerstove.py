"""Compare repeated Cython estimates with exact or independently sampled PokerStove equity."""
import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess

import numpy as np

from poker_eval_faster import estimate_equity_vs_random


SCENARIOS = [
    ("top_pair", ["Ac", "Tc"], ["9c", "2d", "As", "2h", "7h"]),
    ("small_pair", ["5h", "6d"], ["5d", "7h", "Qc", "Qs", "3h"]),
    ("weak_hand", ["4h", "7d"], ["9d", "8c", "9c", "2h", "As"]),
    ("overcards", ["Ks", "Jh"], ["4c", "5d", "5h", "9s", "2c"]),
]


def cases():
    for name, hero, board in SCENARIOS:
        for size in (3, 4, 5):
            for players in (2, 8):
                yield dict(id=f"{name}_{size}_{players}", hero=hero, board=board[:size], players=players, dead=[])
        yield dict(id=f"{name}_river_3", hero=hero, board=board, players=3, dead=[])
    for size in (3, 4, 5):
        yield dict(id=f"flush_draw_{size}_8", hero=["Ah", "Kh"],
                   board=["Qh", "Jh", "2c", "7d", "8s"][:size], players=8, dead=[])
    yield dict(id="board_royal_8", hero=["2c", "3d"], board=["Ah", "Kh", "Qh", "Jh", "Th"], players=8, dead=[])
    yield dict(id="top_pair_flop_10", hero=SCENARIOS[0][1], board=SCENARIOS[0][2][:3], players=10, dead=[])
    yield dict(id="top_pair_flop_dead_8", hero=SCENARIOS[0][1], board=SCENARIOS[0][2][:3], players=8, dead=["Kh", "Qs", "Jc"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Compiled pokerstove_random_reference.cpp")
    parser.add_argument("--revision", required=True, help="PokerStove source revision used to build the reference")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-samples", type=int, default=10_000_000)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    if min(args.reference_samples, args.samples, args.seeds) < 1:
        parser.error("Sample and seed counts must be positive")
    reference = args.reference.resolve()
    payload = dict(source="PokerStove ShowdownEnumerator / independent mt19937_64 + std::shuffle simulation",
                   pokerstove_revision=args.revision, samples=args.samples, seeds=list(range(args.seeds)),
                   reference_samples=args.reference_samples, reference_seed=20260926, cases=[])
    for case in cases():
        exact = case["players"] == 2 or (case["players"] == 3 and len(case["board"]) == 5)
        command = [str(reference), "".join(case["hero"]), "".join(case["board"]), str(case["players"]),
                   "0" if exact else str(args.reference_samples), str(payload["reference_seed"]),
                   "".join(case["dead"]) or "-"]
        baseline = json.loads(subprocess.check_output(command, text=True, timeout=120))
        values = [estimate_equity_vs_random(case["hero"], case["board"], case["players"] - 1,
                                          dead_cards=case["dead"], samples=args.samples, seed=seed).equity
                  for seed in payload["seeds"]]
        errors = [100 * abs(v - baseline["equity"]) for v in values]
        case["reference"] = baseline
        case["comparison"] = dict(mean_abs_error_pp=mean(errors), p95_abs_error_pp=float(np.percentile(errors, 95)),
                                  max_abs_error_pp=max(errors), max_error_seed=errors.index(max(errors)),
                                  mean_signed_error_pp=100 * (mean(values) - baseline["equity"]),
                                  count_over_one_pp=sum(e > 1 for e in errors), first_seed_equity=values[0])
        payload["cases"].append(case)
        print(f"{case['id']}: {baseline['mode']}, equity={100 * baseline['equity']:.5f}%, "
              f"max difference={max(errors):.5f} pp", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
