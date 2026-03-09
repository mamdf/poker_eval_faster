import pytest

# Salta benchmarks si no está instalado el plugin pytest-benchmark
pytest.importorskip("pytest_benchmark")

from poker_eval_faster import (
    evaluate_hands,
    evaluate_one_hand_vs_all,
    evaluate_rank,
    evaluate_ranges
)


@pytest.mark.benchmark(group="evaluate_hands")
def test_benchmark_evaluate_hands(benchmark):
    hands = [['9c', '8c'], ['Tc', 'Td']]
    board = ['Qh', 'Jh', '8s']
    benchmark(lambda: evaluate_hands(hands, board))


@pytest.mark.benchmark(group="evaluate_two_combos_preflop")
def test_benchmark_evaluate_two_combos_preflop(benchmark):
    hands = [['Ac', 'Kc'], ['2d', '2h']]
    benchmark(lambda: evaluate_hands(hands))


@pytest.mark.benchmark(group="evaluate_tree_combos_preflop")
def test_benchmark_evaluate_tree_combos_preflop(benchmark):
    hands = [['Ac', 'Kc'], ['2d', '2h'], ['5s', '6s']]
    benchmark(lambda: evaluate_hands(hands))


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


@pytest.mark.benchmark(group="evaluate_simple_ranges_preflop")
def test_benchmark_evaluate_simple_ranges_preflop(benchmark):
    benchmark(lambda: evaluate_ranges("AA", "KK"))