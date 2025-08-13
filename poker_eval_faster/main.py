from poker_eval_faster import evaluate_one_hand_vs_all_c, hand_to_equity
from poker_eval_faster import evaluate_hands_c, hands_to_equity
from poker_eval_faster import evaluate_c
from typing import Iterable
from typing import List, Tuple
import numpy as np

RANKS_STR = '23456789TJQKA'
SUITS_STR = 'cdhs'
DECK = [r + s for r in RANKS_STR for s in SUITS_STR]
CARDS_TO_INT = {card: i for i, card in enumerate(DECK, start=1)}
RANKING = [None, "NOPAIR", "PAIR", "DOUBLES", "TRIPS", "STRAIGHT", "FLUSH", "FULL", "QUADS", "STRAIGHT_FLUSH"]


def card_to_int(card: str):
    return CARDS_TO_INT[card]


def cards_to_int_array(cards: List[str]):
    return np.array([CARDS_TO_INT[c] for c in cards], dtype='int32')


def cards_to_array(cards: List[int]):
    return np.array(cards, dtype='int32')


def int_to_cards(cards: List[int]):
    return [DECK[i-1] for i in cards]


def ranking_to_category(rank: int) -> Tuple[int, str]:
    rank = rank >> 12  # rank to num category
    return rank, RANKING[rank]


def evaluate_rank(board: List, hand: List = []):
    if type(board[0]) is str:
        cards = cards_to_int_array(hand + board)
    else:
        cards = cards_to_array(hand + board)
    rank = evaluate_c(cards)
    return rank


def evaluate_hands(hands, board=None, eq=True, incomplete_board=False) -> List[float]:
    """
    :param hands: List[List[str, str]] or List[List[int, int]]
    :param board: List[str] or List[int] or None
    :param eq: return equity or combos (win, win... tie, tie...)
    :return: List[float] equity hands or combos
    """
    hands_cards = [card for hand in hands for card in hand]
    if type(hands_cards[0]) is str:  # ex. Ac, Kc
        hands_cards = cards_to_int_array(hands_cards)
    else:  # ex. 45, 50
        hands_cards = cards_to_array(hands_cards)
    len_cards = hands_cards.size

    if board:
        if type(board[0]) is str:  # ex. Ac
            board_cards = cards_to_int_array(board)
        else:  # ex. 45
            board_cards = cards_to_array(board)
        len_cards += board_cards.size
        if np.unique(hands_cards).size + np.unique(board_cards).size != len_cards:  # repeated cards
            ev = np.zeros(len_cards)
        else:
            ev = evaluate_hands_c(hands_cards, board_cards)
    else:
        if np.unique(hands_cards).size != len_cards:   # repeated cards
            ev = np.zeros(len_cards)
        else:
            ev = evaluate_hands_c(hands_cards)

    if eq:
        return hands_to_equity(ev)
    else:
        return list(ev)


def evaluate_one_hand_vs_all(hand, board, eq=True, incomplete_board=False):
    """
    :param hand: List[str, str] or List[int, int]
    :param board: List[str] or List[int]
    :param eq: return equity or combos (win, tie, lose)
    :param incomplete_board: if False and board < 5 cards, complete it with all possible combinations
    :return: List[float] equity hand or combos
    """
    if type(hand[0]) is str:
        cards = cards_to_int_array(hand + board)
    else:
        cards = cards_to_array(hand + board)
    distributions = np.zeros([53, 53])
    ev = evaluate_one_hand_vs_all_c(cards, distributions, incomplete_board)
    if eq:
        result = hand_to_equity(ev)
    else:
        result = list(ev)

    return result


def distribution_one_hand_vs_all(hand, board, sort_distributions=False):
    if type(hand[0]) is str:
        cards = cards_to_int_array(hand + board)
    else:
        cards = cards_to_array(hand + board)
    distributions = np.zeros([53, 53])  # dim 0 is used in flop to fill turn means
    distributions_flop = []
    # create mask for no cards in sort equity
    mask = np.ones(53, dtype='bool')
    mask[cards] = False
    mask[0] = False  # (cards 1-53)
    # evaluate
    ev = evaluate_one_hand_vs_all_c(cards, distributions, incomplete_board=False)

    if len(board) == 3:
        for i in range(1, len(distributions)):  # equity in each turn card (mean rivers)
            if i in cards:  # dist_turn = zeros
                continue
            dist_turn = distributions[i]
            mask_turn = mask.copy()
            mask_turn[i] = False
            valid_cards = dist_turn[mask_turn]
            distributions[0][i] = valid_cards.mean()
            if sort_distributions:
                distributions_flop.append(np.sort(valid_cards))

        if sort_distributions:
            arg_mean = np.argsort(distributions[0][mask])
            return np.array(distributions_flop)[arg_mean]
        else:
            return distributions
    elif len(board) == 4:
        if sort_distributions:
            return np.sort(distributions[0][mask])
        else:
            return distributions[0]
    else:
        return ev


def parse_range_notation(range_str: str) -> List[Tuple[int, int]]:
    """
    Parser de rangos de mano (simplificado pero útil):
    - Pares: "TT", con "+": "TT+" (TT, JJ, QQ, KK, AA)
    - Conjuntos concretos: "AsKs,AdKd"
    - No pares: "AK", "AKs", "AKo"
    - Con "+" en el segundo rango: "A2+" (A2..AK), "A2s+", "K9o+".
    Nota: No soporta aún rangos con "-" ni pesos.
    """
    def rank_index(r: str) -> int:
        return RANKS_STR.index(r)

    def gen_pair_combos(rank: str) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for i, s1 in enumerate(SUITS_STR):
            for j in range(i + 1, len(SUITS_STR)):
                s2 = SUITS_STR[j]
                c1 = CARDS_TO_INT[rank + s1]
                c2 = CARDS_TO_INT[rank + s2]
                out.append((c1, c2))
        return out

    def gen_suited(rank_hi: str, rank_lo: str) -> List[Tuple[int, int]]:
        # Misma pinta para ambos
        out: List[Tuple[int, int]] = []
        for s in SUITS_STR:
            c1 = CARDS_TO_INT[rank_hi + s]
            c2 = CARDS_TO_INT[rank_lo + s]
            out.append((c1, c2))
        return out

    def gen_offsuit(rank_hi: str, rank_lo: str) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for s1 in SUITS_STR:
            for s2 in SUITS_STR:
                if s1 == s2:
                    continue
                out.append((CARDS_TO_INT[rank_hi + s1], CARDS_TO_INT[rank_lo + s2]))
        return out

    def gen_both(rank_hi: str, rank_lo: str) -> List[Tuple[int, int]]:
        return gen_suited(rank_hi, rank_lo) + gen_offsuit(rank_hi, rank_lo)

    tokens = [tok.strip().upper() for tok in range_str.split(',') if tok.strip()]
    result: List[Tuple[int, int]] = []
    seen = set()

    for tok in tokens:
        # Rango con guión (e.g., 66-99, ATs-A2s, K9-KQ)
        if '-' in tok:
            left, right = [part.strip() for part in tok.split('-', 1)]
            # normalizar sufijos y extraer base de cada lado
            def norm(part: str):
                base = part
                qualifier = None
                plus = False
                while True:
                    if base.endswith('+'):
                        plus = True
                        base = base[:-1]
                        continue
                    if base.endswith('S') or base.endswith('O'):
                        qualifier = base[-1]
                        base = base[:-1]
                        continue
                    break
                return base, qualifier, plus

            lbase, lq, lplus = norm(left)
            rbase, rq, rplus = norm(right)
            if lplus or rplus:
                # no se soporta '+' combinado con '-' por ahora
                raise ValueError(f"No se soporta '+' combinado con '-' en: {tok}")
            # pares, e.g. 66-99
            if (
                len(lbase) == 2 and len(rbase) == 2
                and lbase[0] == lbase[1] and rbase[0] == rbase[1]
                and lbase[0] in RANKS_STR and rbase[0] in RANKS_STR
            ):
                start = min(rank_index(lbase[0]), rank_index(rbase[0]))
                end = max(rank_index(lbase[0]), rank_index(rbase[0]))
                for idx in range(start, end + 1):
                    r = RANKS_STR[idx]
                    for combo in gen_pair_combos(r):
                        pair = (combo[0], combo[1]) if combo[0] < combo[1] else (combo[1], combo[0])
                        if pair not in seen:
                            seen.add(pair)
                            result.append(pair)
                continue
            # no pares, e.g. ATs-A2s o K9-KQ (mismo alto en ambos lados)
            if len(lbase) == 2 and len(rbase) == 2 and lbase[0] == rbase[0] and lbase[0] in RANKS_STR and lbase[1] in RANKS_STR and rbase[1] in RANKS_STR:
                hi = lbase[0]
                lo1 = lbase[1]
                lo2 = rbase[1]
                q = lq or rq  # si uno especifica s/o, usamos ese
                start = min(rank_index(lo1), rank_index(lo2))
                end = max(rank_index(lo1), rank_index(lo2))
                for idx in range(start, end + 1):
                    lo = RANKS_STR[idx]
                    if lo == hi:
                        continue
                    if q == 'S':
                        combos = gen_suited(hi, lo)
                    elif q == 'O':
                        combos = gen_offsuit(hi, lo)
                    else:
                        combos = gen_both(hi, lo)
                    for combo in combos:
                        pair = (combo[0], combo[1]) if combo[0] < combo[1] else (combo[1], combo[0])
                        if pair not in seen:
                            seen.add(pair)
                            result.append(pair)
                continue
            raise ValueError(f"Rango con '-' no soportado: {tok}")

        # Casos explícitos de 4 chars, e.g., AsKs
        if len(tok) == 4 and tok[0] in RANKS_STR and tok[2] in RANKS_STR:
            c1 = CARDS_TO_INT[tok[0] + tok[1].lower()]
            c2 = CARDS_TO_INT[tok[2] + tok[3].lower()]
            pair = (c1, c2) if c1 < c2 else (c2, c1)
            if pair not in seen:
                seen.add(pair)
                result.append(pair)
            continue

        # Pares con o sin + (e.g., TT, TT+)
        if len(tok) in (2, 3) and tok[0] == tok[1] and tok[0] in RANKS_STR:
            plus = tok.endswith('+')
            start_idx = rank_index(tok[0])
            end_idx = len(RANKS_STR)  # hasta As
            ranks = RANKS_STR[start_idx:end_idx] if plus else tok[0]
            for r in ranks:
                for combo in gen_pair_combos(r):
                    pair = (combo[0], combo[1]) if combo[0] < combo[1] else (combo[1], combo[0])
                    if pair not in seen:
                        seen.add(pair)
                        result.append(pair)
            continue

        # No pares: AK / AKs / AKo, con opcional + en el segundo rango (A2+, A2s+, A2o+)
        base = tok
        qualifier = None
        plus = False
        # Manejar sufijos en cualquier orden: 'A2S+', 'A2+S', 'A2s+', 'A2+o'
        # Normalizamos a: base sin sufijos, y flags 'qualifier' y 'plus'
        while True:
            if base.endswith('+'):
                plus = True
                base = base[:-1]
                continue
            if base.endswith('S') or base.endswith('O'):
                qualifier = base[-1]
                base = base[:-1]
                continue
            break

        if len(base) == 2 and base[0] in RANKS_STR and base[1] in RANKS_STR and base[0] != base[1]:
            hi = base[0]
            lo_start = base[1]
            lo_candidates = RANKS_STR[rank_index(lo_start):rank_index('A')] if plus else lo_start
            for lo in lo_candidates:
                if hi == lo:
                    continue
                if qualifier == 'S':
                    combos = gen_suited(hi, lo)
                elif qualifier == 'O':
                    combos = gen_offsuit(hi, lo)
                else:
                    combos = gen_both(hi, lo)
                for combo in combos:
                    pair = (combo[0], combo[1]) if combo[0] < combo[1] else (combo[1], combo[0])
                    if pair not in seen:
                        seen.add(pair)
                        result.append(pair)
            continue

        raise ValueError(f"Token de rango no soportado: {tok}")

    return result


def evaluate_ranges(hero_range: Iterable[Tuple[int, int]] | str,
                    villain_range: Iterable[Tuple[int, int]] | str,
                    board=None) -> float:
    """
    Evalúa equity de un rango contra otro.
    Acepta iterables de pares (int,int) o un string simple (ver parse_range_notation).
    """
    from poker_eval_faster import evaluate_range_vs_range_c
    if isinstance(hero_range, str):
        hero_list = parse_range_notation(hero_range)
    else:
        hero_list = list(hero_range)
    if isinstance(villain_range, str):
        villain_list = parse_range_notation(villain_range)
    else:
        villain_list = list(villain_range)
    hero_arr = np.array(hero_list, dtype='int32')
    villain_arr = np.array(villain_list, dtype='int32')
    if board:
        if type(board[0]) is str:
            board_cards = cards_to_int_array(board)
        else:
            board_cards = cards_to_array(board)
    else:
        board_cards = np.array([], dtype='int32')
    return float(__import__('poker_eval_faster').evaluate_range_vs_range_c(hero_arr, villain_arr, board_cards))


if __name__ == '__main__':
    distribution_one_hand_vs_all(['2s', '6s'], ['Qs', 'Ks', '7c'], sort_distributions=True)