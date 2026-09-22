# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from libc.stdint cimport uint32_t, uint64_t

from .main cimport handdat


cdef void _count_board(int hero0, int hero1, int* deck, int n,
                       int turn_index, int river_index, uint32_t board_rank,
                       uint64_t* hero_counts, uint64_t* opponent_counts) noexcept nogil:
    cdef int i, j
    cdef uint32_t prefix, rank
    rank = handdat[handdat[board_rank + hero0] + hero1]
    hero_counts[rank >> 12] += 1
    for i in range(n):
        if i == turn_index or i == river_index:
            continue
        prefix = handdat[board_rank + deck[i]]
        for j in range(i + 1, n):
            if j == turn_index or j == river_index:
                continue
            rank = handdat[prefix + deck[j]]
            opponent_counts[rank >> 12] += 1


cpdef river_category_histograms_c(const int[:] hero, const int[:] board,
                                  const int[:] dead):
    cdef bint blocked[53]
    cdef int deck[52]
    cdef uint64_t hero_counts[10]
    cdef uint64_t opponent_counts[10]
    cdef int i, a, b, card, n = 0, missing = 5 - board.shape[0]
    cdef uint32_t prefix = 53, turn, river
    cdef const int[:] group
    if hero.shape[0] != 2 or board.shape[0] < 3 or board.shape[0] > 5:
        raise ValueError("Expected two hero cards and a flop, turn or river board")
    for i in range(53):
        blocked[i] = False
    for group in (hero, board, dead):
        for i in range(group.shape[0]):
            card = group[i]
            if card < 1 or card > 52:
                raise ValueError("Cards must be deck IDs in 1..52")
            if blocked[card]:
                raise ValueError("Cards must be distinct")
            blocked[card] = True
    for i in range(1, 53):
        if not blocked[i]:
            deck[n] = i
            n += 1
    if n < missing + 2:
        raise ValueError("Not enough cards for the opponent and board runout")
    for i in range(board.shape[0]):
        prefix = handdat[prefix + board[i]]
    for i in range(10):
        hero_counts[i] = opponent_counts[i] = 0
    with nogil:
        if missing == 0:
            _count_board(hero[0], hero[1], deck, n, -1, -1, prefix,
                         hero_counts, opponent_counts)
        else:
            for a in range(n):
                turn = handdat[prefix + deck[a]]
                if missing == 1:
                    _count_board(hero[0], hero[1], deck, n, a, -1, turn,
                                 hero_counts, opponent_counts)
                else:
                    for b in range(a + 1, n):
                        river = handdat[turn + deck[b]]
                        _count_board(hero[0], hero[1], deck, n, a, b, river,
                                     hero_counts, opponent_counts)
    return ([hero_counts[i] for i in range(1, 10)],
            [opponent_counts[i] for i in range(1, 10)])
