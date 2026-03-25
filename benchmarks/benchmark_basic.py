#!/usr/bin/env python3
import argparse
import time
from typing import Callable

from poker_eval_faster import (
    evaluate_hands,
    evaluate_one_hand_vs_all,
    evaluate_rank,
    evaluate_three_way_orders,
    ranking_to_category,
)


def _bench(name: str, func: Callable[[], None], iterations: int) -> None:
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start
    ops_per_sec = iterations / elapsed if elapsed > 0 else float('inf')
    usec_per_op = (elapsed / iterations) * 1e6
    print(f"{name}: {ops_per_sec:,.0f} ops/s  ({usec_per_op:.2f} us/op)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic benchmarks for poker_eval_faster")
    parser.add_argument("--iters", type=int, default=20000, help="Iterations per benchmark")
    args = parser.parse_args()

    # Inputs inspired by unit tests
    hands = [['9c', '8c'], ['Tc', 'Td']]
    board = ['Qh', 'Jh', '8s']
    three_way_hands = [['As', 'Ks'], ['Qh', 'Jh'], ['9c', '9d']]
    three_way_board = ['2c', '7d', 'Th']

    hand = ['Tc', 'Qc']
    board2 = ['Ad', 'Jh', '3s']

    rank_hand = ['Kh', 'Ah']
    rank_board = ['Th', 'Jh', 'Qh']

    print(f"Running {args.iters} iterations per benchmark...\n")

    _bench(
        "evaluate_hands (2 hands on flop)",
        lambda: evaluate_hands(hands, board),
        args.iters,
    )

    _bench(
        "evaluate_one_hand_vs_all (turn)",
        lambda: evaluate_one_hand_vs_all(hand, board2),
        args.iters,
    )

    _bench(
        "evaluate_three_way_orders (3 hands on flop)",
        lambda: evaluate_three_way_orders(three_way_hands, three_way_board),
        args.iters,
    )

    _bench(
        "evaluate_rank (5-7 cards)",
        lambda: (ranking_to_category(evaluate_rank(rank_board, rank_hand))),
        args.iters,
    )


if __name__ == "__main__":
    main()
