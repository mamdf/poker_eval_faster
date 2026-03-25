#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


DEFAULT_PS_EVAL = Path.home() / "pokerstove" / "build" / "bin" / "ps-eval"
OUTPUT_PATTERN = re.compile(
    r"The hand (\S+) has ([0-9.]+) % equity \(([-0-9.]+) ([-0-9.]+) 0 0\)"
)


def _participants_for_case(case: dict) -> list[str]:
    kind = case["kind"]
    if kind == "hands":
        return ["".join(hand) for hand in case["hands"]]
    if kind == "ranges":
        return list(case["ranges"])
    raise ValueError(f"Unsupported case kind: {kind!r}")


def run_ps_eval(ps_eval: Path, participants: list[str], board: list[str]) -> dict[str, object]:
    cmd = [str(ps_eval)]
    for participant in participants:
        cmd += ["-h", participant]
    if board:
        cmd += ["--board", "".join(board)]

    output = subprocess.check_output(cmd, text=True)
    players = []
    total_share = 0.0
    for line in output.strip().splitlines():
        match = OUTPUT_PATTERN.match(line.strip())
        if not match:
            raise ValueError(f"Could not parse ps-eval output line: {line!r}")
        label, _, wins, tie_share = match.groups()
        wins_i = int(round(float(wins)))
        tie_share_f = round(float(tie_share), 12)
        total_share += wins_i + tie_share_f
        players.append(
            {
                "label": label,
                "wins": wins_i,
                "tie_share": tie_share_f,
            }
        )

    total = int(round(total_share))
    for player in players:
        player["equity"] = round((player["wins"] + player["tie_share"]) / total, 12)

    return {
        "players": players,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh 3-way fixture expectations using PokerStove ps-eval.")
    parser.add_argument(
        "--input",
        default="tests/fixtures/three_way_pokerstove_snapshot.json",
        help="Fixture JSON with cases containing hands or ranges.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path. Defaults to overwriting --input.",
    )
    parser.add_argument(
        "--ps-eval",
        default=str(DEFAULT_PS_EVAL),
        help="Path to the PokerStove ps-eval binary.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    ps_eval = Path(args.ps_eval).expanduser()

    payload = json.loads(input_path.read_text())
    for case in payload["cases"]:
        case["expected"] = run_ps_eval(ps_eval, _participants_for_case(case), case.get("board", []))

    payload["source"] = "PokerStove ps-eval exact 3-way snapshots"
    payload["notes"] = (
        "For 3-way, ps-eval exposes solo wins and split-pot equity share, not the full 13-class weak-order "
        "distribution. Range-like cases use explicit comma-separated combo lists, not shorthand like AA/AKo."
    )
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
