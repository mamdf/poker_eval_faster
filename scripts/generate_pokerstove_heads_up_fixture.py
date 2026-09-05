#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import re
import subprocess
from pathlib import Path


DEFAULT_PS_EVAL = os.environ.get("POKERSTOVE_BIN") or shutil.which("ps-eval")
OUTPUT_PATTERN = re.compile(
    r"The hand (\S+) has ([0-9.]+) % equity \(([-0-9.]+) ([-0-9.]+) 0 0\)"
)


def run_ps_eval(ps_eval: Path, hero: list[str], villain: list[str], board: list[str]) -> dict[str, float | int]:
    cmd = [str(ps_eval), "-h", "".join(hero), "-h", "".join(villain)]
    if board:
        cmd += ["--board", "".join(board)]

    output = subprocess.check_output(cmd, text=True)
    parsed = []
    for line in output.strip().splitlines():
        match = OUTPUT_PATTERN.match(line.strip())
        if not match:
            raise ValueError(f"Could not parse ps-eval output line: {line!r}")
        _, equity_pct, wins, tie_half = match.groups()
        parsed.append((float(equity_pct), float(wins), float(tie_half)))

    _, hero_wins, hero_tie_half = parsed[0]
    villain_wins = parsed[1][1]
    villain_tie_half = parsed[1][2]
    total = int(round(hero_wins + villain_wins + hero_tie_half + villain_tie_half))

    return {
        "wins": int(round(hero_wins)),
        "ties": int(round(hero_tie_half * 2)),
        "total": total,
        "equity": round((hero_wins + hero_tie_half) / total, 12),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh HU fixture expectations using PokerStove ps-eval.")
    parser.add_argument(
        "--input",
        default="tests/fixtures/heads_up_counts_snapshot.json",
        help="Fixture JSON with cases containing hero/villain/board.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path. Defaults to overwriting --input.",
    )
    parser.add_argument(
        "--ps-eval",
        default=DEFAULT_PS_EVAL,
        help="PokerStove binary (default: POKERSTOVE_BIN, then ps-eval on PATH).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    if not args.ps_eval:
        parser.error("ps-eval not found: set POKERSTOVE_BIN or pass --ps-eval")
    resolved = shutil.which(os.path.expanduser(args.ps_eval))
    if resolved is None:
        parser.error(f"ps-eval is not executable: {args.ps_eval}")
    ps_eval = Path(resolved).resolve()

    payload = json.loads(input_path.read_text())
    for case in payload["cases"]:
        case["expected"] = run_ps_eval(ps_eval, case["hero"], case["villain"], case["board"])

    payload["source"] = "PokerStove ps-eval exact counts"
    payload["notes"] = (
        "Generated from concrete combo inputs. ps-eval shorthand like AA/AKo/ranges "
        "was observed to crash in this environment, so this script only emits combo-vs-combo cases."
    )
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
