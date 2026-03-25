#!/usr/bin/env python3
import argparse
import time

from poker_eval_faster import (
    build_three_way_class_lookup,
    class_label_to_id,
    multiset_triple_count,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the 3-way 169-class lookup builder on a subset")
    parser.add_argument(
        "--classes",
        default="AA,KK,QQ,AKs,AKo,KQs,KQo",
        help="Comma-separated 169-class labels to include in the subset artifact",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Worker processes to use while building the subset",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Approximate number of entries per scheduling chunk",
    )
    args = parser.parse_args()

    labels = [label.strip() for label in args.classes.split(",") if label.strip()]
    class_indices = [class_label_to_id(label) for label in labels]
    entry_count = multiset_triple_count(len(class_indices))

    start = time.perf_counter()
    artifact = build_three_way_class_lookup(
        class_indices=class_indices,
        processes=args.processes,
        chunk_size=args.chunk_size,
    )
    elapsed = time.perf_counter() - start

    print(f"classes={labels}")
    print(f"entries={entry_count}")
    print(f"legal_entries={artifact.header.legal_count}")
    print(f"elapsed={elapsed * 1e3:.2f} ms")
    print(f"throughput={entry_count / elapsed:.2f} entries/s")


if __name__ == "__main__":
    main()
