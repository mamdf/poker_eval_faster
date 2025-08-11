import pytest

import pytest

# Salta benchmarks si no está instalado el plugin pytest-benchmark
pytest.importorskip("pytest_benchmark")

from poker_eval_faster import (
    evaluate_hands,
    evaluate_one_hand_vs_all,
    evaluate_rank,
)


@pytest.mark.benchmark(group="evaluate_hands")
def test_benchmark_evaluate_hands(benchmark):
    hands = [['9c', '8c'], ['Tc', 'Td']]
    board = ['Qh', 'Jh', '8s']
    benchmark(lambda: evaluate_hands(hands, board))


@pytest.mark.benchmark(group="evaluate_one_vs_all")
def test_benchmark_evaluate_one_hand_vs_all(benchmark):
    hand = ['Tc', 'Qc']
    board = ['Ad', 'Jh', '3s']
    benchmark(lambda: evaluate_one_hand_vs_all(hand, board))


@pytest.mark.benchmark(group="evaluate_rank")
def test_benchmark_evaluate_rank(benchmark):
    hand = ['Kh', 'Ah']
    board = ['Th', 'Jh', 'Qh']
    benchmark(lambda: evaluate_rank(board, hand))
