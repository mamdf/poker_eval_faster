import pytest
from poker_eval_faster import evaluate_rank, ranking_to_category


def test_evaluate_rank_simple():
    rank = evaluate_rank(['Th', 'Jh', 'Qh', 'Kh', 'Ah'], [])
    cat = ranking_to_category(rank)[0]
    assert rank == 36874
    assert cat == 9


def test_evaluate_rank():
    valor_escalera_real = evaluate_rank(["As", "Ks", "Qs", "Js", "Ts", "9c", "8d"])
    valor_poker_3 = evaluate_rank(["3c", "3d", "3h", "3s", "4c", "4d", "4h"])
    valor_poker_2 = evaluate_rank(["2c", "2d", "2h", "2s", "Ac", "Ad", "Ah"])
    valor_full_house = evaluate_rank(["Ac", "Ad", "As", "Kc", "Kd", "5h", "2s"])
    valor_color = evaluate_rank(["Ac", "Kc", "8c", "5c", "2c", "Qh", "Jh"])
    valor_color_2 = evaluate_rank(["Ac", "Kc", "8c", "4c", "2c", "Qh", "Jh"])
    valor_escalera = evaluate_rank(["Ac", "Kc", "Qd", "Jc", "Tc", "9d", "8d"])
    valor_escalera_2 = evaluate_rank(["Ac", "2d", "3c", "4c", "5c", "9d", "8d"])
    valor_par_doble = evaluate_rank(["Ac", "Ad", "Kc", "Kd", "8s", "7h", "3d"])
    assert valor_escalera_real > valor_poker_3
    assert valor_poker_3 > valor_poker_2
    assert valor_poker_2 > valor_full_house
    assert valor_full_house > valor_color
    assert valor_color > valor_color_2
    assert valor_color_2 > valor_escalera
    assert valor_escalera > valor_escalera_2
    assert valor_escalera_2 > valor_par_doble
