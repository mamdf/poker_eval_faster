from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Iterable, List, Sequence, Tuple
import numpy as np

from .eval_cython.hands_evaluate import (
    evaluate_hands_c,
    evaluate_heads_up_counts_c,
    evaluate_range_vs_range_c,
    hands_to_equity,
)
from .eval_cython.main import evaluate_c
from .eval_cython.one_hand_evaluate import evaluate_one_hand_vs_all_c, hand_to_equity
from .eval_cython.three_way_orders import evaluate_three_way_orders_c
from .preflop_canonical import (
    _canonical_preflop_matchup,
    _clear_preflop_canonical_caches,
    _combo_ids_from_array,
    _evaluate_ranges_preflop_cached,
    _preflop_canonical_counts,
    _preflop_range_matchup_profile,
    canonical_combo_masks,
    canonical_combos,
)

RANKS_STR = '23456789TJQKA'
SUITS_STR = 'cdhs'
DECK = [r + s for r in RANKS_STR for s in SUITS_STR]
CARDS_TO_INT = {card: i for i, card in enumerate(DECK, start=1)}
RANKING = [None, "NOPAIR", "PAIR", "DOUBLES", "TRIPS", "STRAIGHT", "FLUSH", "FULL", "QUADS", "STRAIGHT_FLUSH"]
THREE_WAY_ORDER_LABELS = (
    "A>B>C",
    "A>C>B",
    "B>A>C",
    "B>C>A",
    "C>A>B",
    "C>B>A",
    "A=B>C",
    "A=C>B",
    "B=C>A",
    "A>B=C",
    "B>A=C",
    "C>A=B",
    "A=B=C",
)
_THREE_WAY_SOLO_WIN_INDICES = (
    (0, 1, 9),
    (2, 3, 10),
    (4, 5, 11),
)
_THREE_WAY_TWO_WAY_TOP_TIE_INDICES = (
    (6, 7),
    (6, 8),
    (7, 8),
)
_THREE_WAY_EQUITY_WEIGHTS = (
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
)


@dataclass(frozen=True)
class HeadsUpCounts:
    wins: int
    ties: int
    total: int

    @property
    def losses(self) -> int:
        return max(self.total - self.wins - self.ties, 0)

    @property
    def equity(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.wins + (self.ties / 2.0)) / self.total


@dataclass(frozen=True)
class ThreeWayTopCounts:
    wins: int
    two_way_ties: int
    three_way_ties: int
    total: int

    @property
    def ties(self) -> int:
        return self.two_way_ties + self.three_way_ties

    @property
    def equity(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.wins + (self.two_way_ties / 2.0) + (self.three_way_ties / 3.0)) / self.total


@dataclass(frozen=True)
class ThreeWayOrderCounts:
    order_counts: Tuple[int, ...]
    total: int

    def __post_init__(self) -> None:
        if len(self.order_counts) != len(THREE_WAY_ORDER_LABELS):
            raise ValueError(
                f"ThreeWayOrderCounts expects {len(THREE_WAY_ORDER_LABELS)} order counts, "
                f"got {len(self.order_counts)}"
            )

    @property
    def equities(self) -> Tuple[float, float, float]:
        if self.total == 0:
            return 0.0, 0.0, 0.0

        equity_a = 0.0
        equity_b = 0.0
        equity_c = 0.0
        for count, weights in zip(self.order_counts, _THREE_WAY_EQUITY_WEIGHTS):
            if count == 0:
                continue
            equity_a += count * weights[0]
            equity_b += count * weights[1]
            equity_c += count * weights[2]
        return (
            equity_a / self.total,
            equity_b / self.total,
            equity_c / self.total,
        )

    def first_place_counts(self, player_idx: int) -> ThreeWayTopCounts:
        if player_idx not in (0, 1, 2):
            raise ValueError(f"Player index must be 0, 1 or 2, got {player_idx}")

        wins = sum(self.order_counts[idx] for idx in _THREE_WAY_SOLO_WIN_INDICES[player_idx])
        two_way_ties = sum(self.order_counts[idx] for idx in _THREE_WAY_TWO_WAY_TOP_TIE_INDICES[player_idx])
        return ThreeWayTopCounts(
            wins=wins,
            two_way_ties=two_way_ties,
            three_way_ties=self.order_counts[12],
            total=self.total,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            label: int(count)
            for label, count in zip(THREE_WAY_ORDER_LABELS, self.order_counts)
        }


@dataclass(frozen=True)
class HeadsUpLookupTable:
    combo_indices: np.ndarray
    combo_cards: np.ndarray
    win_tie_counts: np.ndarray
    valid_mask: np.ndarray
    board: Tuple[int, ...]
    total: int

    def pair_index(self, first_idx: int, second_idx: int) -> int:
        local_a = int(np.where(self.combo_indices == first_idx)[0][0])
        local_b = int(np.where(self.combo_indices == second_idx)[0][0])
        if local_a == local_b:
            raise ValueError("A lookup pair needs two distinct combo ids.")
        return packed_pair_index(local_a, local_b, self.combo_indices.size)

    def counts_for_ids(self, first_idx: int, second_idx: int) -> HeadsUpCounts:
        pair_idx = self.pair_index(first_idx, second_idx)
        if not bool(self.valid_mask[pair_idx]):
            return HeadsUpCounts(0, 0, 0)

        counts = self.win_tie_counts[pair_idx]
        if first_idx < second_idx:
            wins = int(counts[0])
        else:
            wins = self.total - int(counts[0]) - int(counts[1])
        return HeadsUpCounts(wins=wins, ties=int(counts[1]), total=self.total)


def card_to_int(card: str):
    return CARDS_TO_INT[card]


def cards_to_int_array(cards: List[str]):
    return np.array([CARDS_TO_INT[c] for c in cards], dtype='int32')


def cards_to_array(cards: List[int]):
    return np.array(cards, dtype='int32')


def int_to_cards(cards: List[int]):
    return [DECK[i-1] for i in cards]


def combo_id_to_cards(combo_id: int) -> List[str]:
    combo = canonical_combos()[int(combo_id)]
    return int_to_cards(combo.tolist())


def combo_id_to_str(combo_id: int) -> str:
    return "".join(combo_id_to_cards(combo_id))


def _cards_are_strings(cards: Sequence) -> bool:
    return len(cards) > 0 and isinstance(cards[0], str)


def _normalize_cards(cards: Sequence) -> np.ndarray:
    if _cards_are_strings(cards):
        return cards_to_int_array(list(cards))
    return cards_to_array(list(cards))


def _normalize_board(board) -> np.ndarray:
    if board is None:
        return np.array([], dtype='int32')
    return _normalize_cards(board)


def _cards_have_duplicates(cards: np.ndarray) -> bool:
    return np.unique(cards).size != cards.size


def _cards_mask(cards: Sequence[int]) -> np.uint64:
    mask = np.uint64(0)
    for card in cards:
        mask |= np.uint64(1) << np.uint64(int(card) - 1)
    return mask


def _normalize_combo(combo: Sequence) -> np.ndarray:
    if len(combo) != 2:
        raise ValueError(f"A combo must contain exactly 2 cards, got {combo!r}")
    return _normalize_cards(combo)


@lru_cache(maxsize=256)
def _range_combo_ids_from_string(range_str: str) -> Tuple[int, ...]:
    return _combo_ids_from_array(_range_array_from_string(range_str))


def _clear_preflop_caches() -> None:
    _parse_range_notation_cached.cache_clear()
    _range_array_from_string.cache_clear()
    _range_combo_ids_from_string.cache_clear()
    _clear_preflop_canonical_caches()


def packed_pair_index(first_idx: int, second_idx: int, num_items: int) -> int:
    if first_idx == second_idx:
        raise ValueError("A packed pair index needs two distinct items.")
    if first_idx > second_idx:
        first_idx, second_idx = second_idx, first_idx
    return (first_idx * num_items) - ((first_idx * (first_idx + 1)) // 2) + (second_idx - first_idx - 1)


def combo_to_hand_class(combo: Sequence[int] | Sequence[str]) -> str:
    combo_arr = _normalize_combo(combo)
    first = int(combo_arr[0]) - 1
    second = int(combo_arr[1]) - 1
    rank_one = RANKS_STR[first // len(SUITS_STR)]
    suit_one = SUITS_STR[first % len(SUITS_STR)]
    rank_two = RANKS_STR[second // len(SUITS_STR)]
    suit_two = SUITS_STR[second % len(SUITS_STR)]

    if rank_one == rank_two:
        return rank_one + rank_two

    idx_one = RANKS_STR.index(rank_one)
    idx_two = RANKS_STR.index(rank_two)
    if idx_one > idx_two:
        hi_rank, hi_suit = rank_one, suit_one
        lo_rank, lo_suit = rank_two, suit_two
    else:
        hi_rank, hi_suit = rank_two, suit_two
        lo_rank, lo_suit = rank_one, suit_one
    suffix = 's' if hi_suit == lo_suit else 'o'
    return hi_rank + lo_rank + suffix


def ranking_to_category(rank: int) -> Tuple[int, str]:
    rank = rank >> 12  # rank to num category
    return rank, RANKING[rank]


def evaluate_rank(board: List, hand: List = []):
    cards = _normalize_cards(list(hand) + list(board))
    return evaluate_c(cards)


def evaluate_hands(hands, board=None, eq=True, incomplete_board=False) -> List[float]:
    """
    :param hands: List[List[str, str]] or List[List[int, int]]
    :param board: List[str] or List[int] or None
    :param eq: return equity or combos (win, win... tie, tie...)
    :return: List[float] equity hands or combos
    """
    hands_cards = _normalize_cards([card for hand in hands for card in hand])
    len_cards = hands_cards.size

    if board:
        board_cards = _normalize_cards(board)
        len_cards += board_cards.size
        if _cards_have_duplicates(np.concatenate((hands_cards, board_cards))):
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
    cards = _normalize_cards(list(hand) + list(board))
    if _cards_have_duplicates(cards):
        return 0.0 if eq else [0.0, 0.0, 0.0]
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


def evaluate_heads_up_counts(hero_hand, villain_hand, board=None) -> HeadsUpCounts:
    hero_cards = _normalize_combo(hero_hand)
    villain_cards = _normalize_combo(villain_hand)
    board_cards = _normalize_board(board)
    all_cards = np.concatenate((hero_cards, villain_cards, board_cards))
    if _cards_have_duplicates(all_cards):
        return HeadsUpCounts(0, 0, 0)

    counts = evaluate_heads_up_counts_c(hero_cards, villain_cards, board_cards)
    return HeadsUpCounts(wins=int(counts[0]), ties=int(counts[1]), total=int(counts[2]))


def evaluate_three_way_orders(hands, board=None) -> ThreeWayOrderCounts:
    if len(hands) != 3:
        raise ValueError(f"Three-way evaluation expects exactly 3 hands, got {len(hands)}")

    normalized_hands = [_normalize_combo(hand) for hand in hands]
    hands_cards = np.concatenate(normalized_hands).astype('int32', copy=False)
    board_cards = _normalize_board(board)
    all_cards = np.concatenate((hands_cards, board_cards))
    if _cards_have_duplicates(all_cards):
        return ThreeWayOrderCounts(
            order_counts=(0,) * len(THREE_WAY_ORDER_LABELS),
            total=0,
        )

    counts = evaluate_three_way_orders_c(hands_cards, board_cards)
    order_counts = tuple(int(value) for value in counts.tolist())
    return ThreeWayOrderCounts(order_counts=order_counts, total=int(sum(order_counts)))


def build_heads_up_lookup(board=None, combo_indices: Iterable[int] | None = None) -> HeadsUpLookupTable:
    board_cards = _normalize_board(board)
    if _cards_have_duplicates(board_cards):
        raise ValueError("Board cards must be unique.")

    combos = canonical_combos()
    masks = canonical_combo_masks()
    if combo_indices is None:
        selected = np.arange(combos.shape[0], dtype=np.int32)
    else:
        selected = np.array(sorted(set(int(idx) for idx in combo_indices)), dtype=np.int32)

    num_selected = selected.size
    num_pairs = (num_selected * (num_selected - 1)) // 2
    win_tie_counts = np.zeros((num_pairs, 2), dtype=np.uint32)
    valid_mask = np.zeros(num_pairs, dtype=bool)
    board_mask = _cards_mask(board_cards)
    total = comb(48 - board_cards.size, 5 - board_cards.size)

    for local_first, combo_first_idx in enumerate(selected):
        first_mask = int(masks[int(combo_first_idx)])
        if first_mask & int(board_mask):
            continue
        for local_second in range(local_first + 1, num_selected):
            combo_second_idx = int(selected[local_second])
            second_mask = int(masks[combo_second_idx])
            if second_mask & int(board_mask):
                continue
            pair_idx = packed_pair_index(local_first, local_second, num_selected)
            if first_mask & second_mask:
                continue

            counts = evaluate_heads_up_counts_c(combos[int(combo_first_idx)], combos[combo_second_idx], board_cards)
            win_tie_counts[pair_idx, 0] = counts[0]
            win_tie_counts[pair_idx, 1] = counts[1]
            valid_mask[pair_idx] = True

    return HeadsUpLookupTable(
        combo_indices=selected,
        combo_cards=combos[selected],
        win_tie_counts=win_tie_counts,
        valid_mask=valid_mask,
        board=tuple(int(card) for card in board_cards.tolist()),
        total=total,
    )


def aggregate_heads_up_lookup_by_class(lookup: HeadsUpLookupTable) -> dict[Tuple[str, str], HeadsUpCounts]:
    aggregated: dict[Tuple[str, str], list[int]] = {}

    for local_first, combo_first_idx in enumerate(lookup.combo_indices):
        hero_class = combo_to_hand_class(lookup.combo_cards[local_first])
        for local_second in range(local_first + 1, lookup.combo_indices.size):
            pair_idx = packed_pair_index(local_first, local_second, lookup.combo_indices.size)
            if not bool(lookup.valid_mask[pair_idx]):
                continue

            villain_class = combo_to_hand_class(lookup.combo_cards[local_second])
            counts = lookup.win_tie_counts[pair_idx]
            key = (hero_class, villain_class)
            bucket = aggregated.setdefault(key, [0, 0, 0])
            bucket[0] += int(counts[0])
            bucket[1] += int(counts[1])
            bucket[2] += lookup.total

            reverse_key = (villain_class, hero_class)
            reverse_bucket = aggregated.setdefault(reverse_key, [0, 0, 0])
            reverse_bucket[0] += lookup.total - int(counts[0]) - int(counts[1])
            reverse_bucket[1] += int(counts[1])
            reverse_bucket[2] += lookup.total

    return {
        key: HeadsUpCounts(wins=value[0], ties=value[1], total=value[2])
        for key, value in aggregated.items()
    }


@lru_cache(maxsize=256)
def _parse_range_notation_cached(range_str: str) -> Tuple[Tuple[int, int], ...]:
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

    return tuple(result)


def parse_range_notation(range_str: str) -> List[Tuple[int, int]]:
    return list(_parse_range_notation_cached(range_str))


@lru_cache(maxsize=256)
def _range_array_from_string(range_str: str) -> np.ndarray:
    return np.array(_parse_range_notation_cached(range_str), dtype='int32')


def evaluate_ranges(hero_range: Iterable[Tuple[int, int]] | str,
                    villain_range: Iterable[Tuple[int, int]] | str,
                    board=None) -> float:
    """
    Evalúa equity de un rango contra otro.
    Acepta iterables de pares (int,int) o un string simple (ver parse_range_notation).
    """
    hero_combo_ids: Tuple[int, ...] | None = None
    villain_combo_ids: Tuple[int, ...] | None = None
    hero_arr: np.ndarray | None = None
    villain_arr: np.ndarray | None = None
    board_cards = _normalize_board(board)
    if isinstance(hero_range, str):
        if board_cards.size == 0:
            hero_combo_ids = _range_combo_ids_from_string(hero_range)
        else:
            hero_arr = _range_array_from_string(hero_range)
    else:
        hero_arr = np.array(list(hero_range), dtype='int32')
    if isinstance(villain_range, str):
        if board_cards.size == 0:
            villain_combo_ids = _range_combo_ids_from_string(villain_range)
        else:
            villain_arr = _range_array_from_string(villain_range)
    else:
        villain_arr = np.array(list(villain_range), dtype='int32')
    if board_cards.size == 0:
        if hero_combo_ids is None:
            if hero_arr is None:
                hero_arr = _range_array_from_string(hero_range)
            hero_combo_ids = _combo_ids_from_array(hero_arr)
        if villain_combo_ids is None:
            if villain_arr is None:
                villain_arr = _range_array_from_string(villain_range)
            villain_combo_ids = _combo_ids_from_array(villain_arr)
        return _evaluate_ranges_preflop_cached(hero_combo_ids, villain_combo_ids)
    if hero_arr is None:
        hero_arr = _range_array_from_string(hero_range)
    if villain_arr is None:
        villain_arr = _range_array_from_string(villain_range)
    return float(evaluate_range_vs_range_c(hero_arr, villain_arr, board_cards))


if __name__ == '__main__':
    distribution_one_hand_vs_all(['2s', '6s'], ['Qs', 'Ks', '7c'], sort_distributions=True)
