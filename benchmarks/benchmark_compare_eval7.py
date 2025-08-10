#!/usr/bin/env python3
import argparse
import random
import time
from typing import Callable, List, Optional, Tuple

from poker_eval_faster import evaluate_rank, ranking_to_category

# Local deck representation (strings)
RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK_STR = [r + s for r in RANKS for s in SUITS]


def chunk_seven(cards: List[str]) -> Tuple[List[str], List[str]]:
    """Split 7 cards into (hand2, board5) for our API."""
    hand = cards[:2]
    board = cards[2:]
    return hand, board


def timeit(name: str, func: Callable[[], None], iterations: int) -> Tuple[str, float, float]:
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start
    ops_per_sec = iterations / elapsed if elapsed > 0 else float('inf')
    usec_per_op = (elapsed / iterations) * 1e6
    print(f"{name}: {ops_per_sec:,.0f} ops/s  ({usec_per_op:.2f} us/op)")
    return name, ops_per_sec, usec_per_op


# ---- Our evaluator wrapper (7-card rank via evaluate_rank) ----

def make_our_eval_fn(samples: List[List[str]]) -> Callable[[], None]:
    i = 0
    n = len(samples)

    def run_once() -> None:
        nonlocal i
        seven = samples[i]
        i = (i + 1) % n
        hand, board = chunk_seven(seven)
        _ = ranking_to_category(evaluate_rank(board, hand))

    return run_once


# ---- eval7 wrapper ----

def try_import_eval7():
    try:
        import eval7  # type: ignore
        return eval7
    except Exception:
        return None


def make_eval7_eval_fn(samples: List[List[str]], eval7_mod) -> Callable[[], None]:
    i = 0
    n = len(samples)

    def to_eval7(cards: List[str]):
        return [eval7_mod.Card(c) for c in cards]

    def run_once() -> None:
        nonlocal i
        seven = samples[i]
        i = (i + 1) % n
        cards = to_eval7(seven)
        _ = eval7_mod.evaluate(cards)

    return run_once


# ---- pyeval7 wrapper (best-effort; falls back to eval7-like API if available) ----

def try_import_pyeval7():
    try:
        import pyeval7  # type: ignore
        return pyeval7
    except Exception:
        return None


def make_pyeval7_eval_fn(samples: List[List[str]], pe7_mod) -> Optional[Callable[[], None]]:
    i = 0
    n = len(samples)

    # Try eval7-compatible API first (Card + evaluate)
    if hasattr(pe7_mod, "Card") and hasattr(pe7_mod, "evaluate"):
        def to_cards(cards: List[str]):
            return [pe7_mod.Card(c) for c in cards]

        def run_once() -> None:
            nonlocal i
            seven = samples[i]
            i = (i + 1) % n
            _ = pe7_mod.evaluate(to_cards(seven))

        return run_once

    # Otherwise, try a common rank function name over strings
    candidate_funcs = [
        "evaluate_7cards",
        "rank_7cards",
        "rank_7",
        "evaluate",
    ]
    for fname in candidate_funcs:
        fn = getattr(pe7_mod, fname, None)
        if callable(fn):
            def run_once(fn=fn):  # bind fn
                nonlocal i
                seven = samples[i]
                i = (i + 1) % n
                _ = fn(seven)
            return run_once

    return None


def generate_samples(num_samples: int) -> List[List[str]]:
    rng = random.Random(1337)
    samples: List[List[str]] = []
    for _ in range(num_samples):
        samples.append(rng.sample(DECK_STR, 7))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 7-card ranking throughput vs eval7/pyeval7")
    parser.add_argument("--iters", type=int, default=20000, help="Iterations per implementation")
    parser.add_argument("--samples", type=int, default=1000, help="Unique random 7-card samples to cycle")
    args = parser.parse_args()

    samples = generate_samples(args.samples)

    print(f"Comparing 7-card ranking throughput (higher ops/s is better)\n")

    results: List[Tuple[str, float, float]] = []

    # Ours
    results.append(timeit("poker_eval_faster (7-card via evaluate_rank)", make_our_eval_fn(samples), args.iters))

    # eval7
    eval7_mod = try_import_eval7()
    if eval7_mod is not None:
        results.append(timeit("eval7", make_eval7_eval_fn(samples, eval7_mod), args.iters))
    else:
        print("eval7 not installed. Install with: pip install eval7")

    # pyeval7
    pe7_mod = try_import_pyeval7()
    if pe7_mod is not None:
        pe7_fn = make_pyeval7_eval_fn(samples, pe7_mod)
        if pe7_fn is not None:
            results.append(timeit("pyeval7", pe7_fn, args.iters))
        else:
            print("pyeval7 installed but API not recognized for 7-card ranking; skipping.")
    else:
        print("pyeval7 not installed. Install with: pip install pyeval7")

    # simple summary
    if results:
        print("\nSummary (ops/s):")
        for name, ops, _ in results:
            print(f"- {name}: {ops:,.0f}")


if __name__ == "__main__":
    main()
