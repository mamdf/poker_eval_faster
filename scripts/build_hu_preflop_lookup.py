from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_eval_faster.hu_lookup import write_hu_preflop_lookup


def _parse_combo_ids(raw_ids: str | None, limit: int | None) -> list[int] | None:
    if raw_ids:
        return [int(token.strip()) for token in raw_ids.split(",") if token.strip()]
    if limit is not None:
        return list(range(limit))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a preflop HU combo-vs-combo lookup artifact.")
    parser.add_argument("output", type=Path, help="Output artifact path.")
    parser.add_argument(
        "--combo-ids",
        help="Comma-separated canonical combo ids to include. Defaults to all 1326 combos.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Build only the first N canonical combos. Useful for smoke tests.",
    )
    args = parser.parse_args()

    combo_indices = _parse_combo_ids(args.combo_ids, args.limit)
    summary = write_hu_preflop_lookup(args.output, combo_indices=combo_indices)
    print(
        f"Wrote {summary.path} with {summary.combo_count} combos, "
        f"{summary.legal_count} legal pairs and {summary.entry_count} triangular entries."
    )


if __name__ == "__main__":
    main()
