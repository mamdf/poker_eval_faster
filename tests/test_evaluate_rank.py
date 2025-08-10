from poker_eval_faster import evaluate_rank, ranking_to_category


def test_evaluate_rank_simple():
    rank = evaluate_rank(['Th', 'Jh', 'Qh', 'Kh', 'Ah'], [])
    cat = ranking_to_category(rank)[0]
    assert rank == 36874
    assert cat == 9


