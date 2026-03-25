import pytest

# Salta benchmarks si no está instalado el plugin pytest-benchmark
pytest.importorskip("pytest_benchmark")

from poker_eval_faster import (
    evaluate_hands,
    evaluate_one_hand_vs_all,
    evaluate_rank,
    evaluate_ranges,
    evaluate_heads_up_counts,
    evaluate_three_way_orders,
    evaluate_three_way_ranges,
)
from poker_eval_faster.main import (
    _clear_preflop_caches,
    _parse_range_notation_cached,
    parse_range_notation,
)


COMPLEX_HERO_RANGE = "99+,AQs+,AQo+"
COMPLEX_VILLAIN_RANGE = "JJ-77,AQs-A9s,KJs+,QJs,AQo-AJo,KQo"
THREE_WAY_RANGE_A = "AKs,AKo,QQ,JJ"
THREE_WAY_RANGE_B = "TT-99,AQs-AJs,KQs"
THREE_WAY_RANGE_C = "88-77,ATs-A8s,KJs+,QJs,AJo"


def _benchmark_cold_range(hero: str, villain: str) -> None:
    _clear_preflop_caches()
    evaluate_ranges(hero, villain)


def _benchmark_cold_three_way_range(range_a: str, range_b: str, range_c: str) -> None:
    _clear_preflop_caches()
    evaluate_three_way_ranges(range_a, range_b, range_c)


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


@pytest.mark.benchmark(group="evaluate_three_way_orders_preflop")
def test_benchmark_evaluate_three_way_orders_preflop(benchmark):
    hands = [['Ac', 'Kc'], ['2d', '2h'], ['5s', '6s']]
    benchmark(lambda: evaluate_three_way_orders(hands))


@pytest.mark.benchmark(group="evaluate_three_way_orders_flop")
def test_benchmark_evaluate_three_way_orders_flop(benchmark):
    hands = [['As', 'Ks'], ['Qh', 'Jh'], ['9c', '9d']]
    board = ['2c', '7d', 'Th']
    benchmark(lambda: evaluate_three_way_orders(hands, board))


@pytest.mark.benchmark(group="evaluate_three_way_ranges_preflop_warm")
def test_benchmark_evaluate_three_way_ranges_preflop_warm(benchmark):
    evaluate_three_way_ranges(THREE_WAY_RANGE_A, THREE_WAY_RANGE_B, THREE_WAY_RANGE_C)
    benchmark(lambda: evaluate_three_way_ranges(THREE_WAY_RANGE_A, THREE_WAY_RANGE_B, THREE_WAY_RANGE_C))


@pytest.mark.benchmark(group="evaluate_three_way_ranges_preflop_cold")
def test_benchmark_evaluate_three_way_ranges_preflop_cold(benchmark):
    benchmark(lambda: _benchmark_cold_three_way_range(THREE_WAY_RANGE_A, THREE_WAY_RANGE_B, THREE_WAY_RANGE_C))


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
    evaluate_ranges("AA", "KK")
    benchmark(lambda: evaluate_ranges("AA", "KK"))


@pytest.mark.benchmark(group="evaluate_simple_ranges_preflop_cold")
def test_benchmark_evaluate_simple_ranges_preflop_cold(benchmark):
    benchmark(lambda: _benchmark_cold_range("AA", "KK"))


@pytest.mark.benchmark(group="evaluate_evaluate_heads_up_counts")
def test_benchmark_evaluate_heads_up_counts(benchmark):
    benchmark(lambda: evaluate_heads_up_counts(['As', 'Ah'], ['Ks', 'Kh'], []))


@pytest.mark.benchmark(group="parse_simple_range_cold")
def test_benchmark_parse_simple_range_cold(benchmark):
    def run():
        _parse_range_notation_cached.cache_clear()
        parse_range_notation("AA")

    benchmark(run)


@pytest.mark.benchmark(group="parse_complex_range_cold")
def test_benchmark_parse_complex_range_cold(benchmark):
    def run():
        _parse_range_notation_cached.cache_clear()
        parse_range_notation("JJ+,AQs+,AQo+")

    benchmark(run)


@pytest.mark.benchmark(group="evaluate_complex_ranges_preflop_warm")
def test_benchmark_evaluate_complex_ranges_preflop_warm(benchmark):
    evaluate_ranges(COMPLEX_HERO_RANGE, COMPLEX_VILLAIN_RANGE)
    benchmark(lambda: evaluate_ranges(COMPLEX_HERO_RANGE, COMPLEX_VILLAIN_RANGE))


@pytest.mark.benchmark(group="evaluate_complex_ranges_preflop_cold")
def test_benchmark_evaluate_complex_ranges_preflop_cold(benchmark):
    benchmark(lambda: _benchmark_cold_range(COMPLEX_HERO_RANGE, COMPLEX_VILLAIN_RANGE))
