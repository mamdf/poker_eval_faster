import click
import numpy as np
from poker_eval_faster import evaluate_hands, evaluate_one_hand_vs_all, hand_to_equity, hands_to_equity
from poker_eval_faster import evaluate_rank, ranking_to_category


def parser_evaluate_hands(hands, board, incomplete_board):
    # evaluate and parser results
    res_combos = evaluate_hands(hands, board, eq=False, incomplete_board=incomplete_board)
    equities = hands_to_equity(np.array(res_combos))
    combos_hands = [[res_combos[i], res_combos[len(hands) * 2 - 1 - i]] for i in range(len(hands))]
    # print to console
    for i, eq in enumerate(equities):
        click.echo(f"The hand {''.join(hands[i])} has {round(eq * 100, 4)} % equity "
                   f"{combos_hands[i][0], combos_hands[i][1]}")


def parser_evaluate_one_hand_vs_all(hand, board, incomplete_board):
    res_combos = evaluate_one_hand_vs_all(hand, board, eq=False, incomplete_board=incomplete_board)
    eq: float = hand_to_equity(np.array(res_combos))
    click.echo(f"The hand {''.join(hand)} vs all hands has {round(eq * 100, 4)} % equity "
               f"{tuple(res_combos)}")


def parser_evaluate_rank(hand, board):
    rank = evaluate_rank(board, hand)
    category = ranking_to_category(rank)
    click.echo(f"The hand is ranking {rank}, category {category[0]}: {category[1]}")


@click.command()
@click.argument('hands', nargs=-1)
@click.option('--board', default='', help='board cards to evaluate, Ex. AcQc9d')
@click.option('-i', '--incomplete-board', default=False, is_flag=True,
              help='evaluate on the current board, no autocomplete')
@click.option('-e', '--evaluate', default=False, is_flag=True,
              help='evaluate only the rank hand (no vs hands). working just with one hand')
def run(hands: str, board: str, incomplete_board: bool, evaluate: bool):
    """
    :arg hands: to evaluate, separate by whitespace (ex: AcKc QdQh)
    """
    hands = [[i[:2], i[2:]] for i in hands]
    board = [board[i] + board[i + 1] for i in range(0, len(board), 2)]
    # select eval function and args
    if len(hands) > 1:
        parser_evaluate_hands(hands, board, incomplete_board)
    else:
        if evaluate:
            parser_evaluate_rank(hands[0], board)
        else:
            parser_evaluate_one_hand_vs_all(hands[0], board, incomplete_board)


if __name__ == '__main__':
    run()
