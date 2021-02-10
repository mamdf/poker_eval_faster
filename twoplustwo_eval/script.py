import click
import numpy as np
from twoplustwo_eval import evaluate_hands, evaluate_one_hand_vs_all, hand_to_equity, hands_to_equity


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


@click.command()
@click.argument('hands', nargs=-1)
@click.option('--board', default='', help='board cards to evaluate, Ex. AcQc9d')
@click.option('-i', '--incomplete-board', default=False, is_flag=True,
              help='evaluate on the current board, no autocomplete')
def run(hands: str, board: str, incomplete_board: bool):
    """
    :arg hands: to evaluate, separate by whitespace (ex: AcKc QdQh)
    """
    hands = [[i[:2], i[2:]] for i in hands]
    board = [board[i] + board[i + 1] for i in range(0, len(board), 2)]
    # select eval function and args
    if len(hands) > 1:
        parser_evaluate_hands(hands, board, incomplete_board)
    else:
        parser_evaluate_one_hand_vs_all(hands[0], board, incomplete_board)


if __name__ == '__main__':
    run()
