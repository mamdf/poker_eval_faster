# cython: boundscheck=False, wraparound=False, cdivision=True
"""Uniform postflop Monte Carlo with per-call state and split-pot accounting."""
from libc.stdint cimport uint32_t, uint64_t
from libc.string cimport memcpy

from poker_eval_faster.eval_cython.main cimport handdat


# PCG32 helpers adapted to Cython from pcg-c-basic (stream 54):
# https://github.com/imneme/pcg-c-basic/blob/master/pcg_basic.c
# Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>.
# Licensed under Apache-2.0; see PCG-LICENSE.txt. Changes: local uint64 state,
# fixed stream, and precomputed rejection thresholds for bounded draws.
cdef inline uint32_t _pcg32(uint64_t* state) noexcept nogil:
    cdef uint64_t old = state[0]
    cdef uint32_t shifted = <uint32_t>(((old >> 18) ^ old) >> 27)
    cdef uint32_t rotation = <uint32_t>(old >> 59)
    state[0] = old * <uint64_t>6364136223846793005 + <uint64_t>109
    return (shifted >> rotation) | (shifted << ((-rotation) & 31))


cdef inline uint32_t _bounded(uint64_t* state, uint32_t bound,
                              uint32_t threshold) noexcept nogil:
    cdef uint32_t value = _pcg32(state)
    while value < threshold:
        value = _pcg32(state)
    return value % bound


def estimate_equity_vs_random_c(int[::1] hero, int[::1] board,
                                int[::1] dead, int num_opponents,
                                uint64_t samples, uint64_t seed):
    """Internal kernel: inputs must first be validated by the Python API."""
    cdef bint blocked[53]
    cdef int base[52]
    cdef int deck[52]
    cdef uint32_t thresholds[23]
    cdef uint64_t outcomes[11]
    cdef uint64_t state = 0, sample, ties = 0
    cdef int i, j, tmp, n = 0, winners, offset
    cdef int missing = 5 - board.shape[0]
    cdef int draws = missing + 2 * num_opponents
    cdef uint32_t prefix = 53, final_board, hero_rank, rival_rank
    cdef double pot_share = 0.0

    for i in range(53):
        blocked[i] = False
    blocked[hero[0]] = blocked[hero[1]] = True
    for i in range(board.shape[0]):
        blocked[board[i]] = True
        prefix = handdat[prefix + board[i]]
    for i in range(dead.shape[0]):
        blocked[dead[i]] = True
    for i in range(1, 53):
        if not blocked[i]:
            base[n] = i
            n += 1
    for i in range(draws):
        thresholds[i] = <uint32_t>(-<uint32_t>(n - i)) % <uint32_t>(n - i)
    for i in range(11):
        outcomes[i] = 0

    with nogil:
        _pcg32(&state)
        state += seed
        _pcg32(&state)
        for sample in range(samples):
            memcpy(deck, base, n * sizeof(int))
            # Draw every card before comparing ranks, so early losses do not
            # alter the random stream or the distribution of later deals.
            for i in range(draws):
                j = i + <int>_bounded(&state, n - i, thresholds[i])
                tmp = deck[i]
                deck[i] = deck[j]
                deck[j] = tmp
            final_board = prefix
            for i in range(missing):
                final_board = handdat[final_board + deck[i]]
            hero_rank = handdat[handdat[final_board + hero[0]] + hero[1]]
            winners = 1
            for i in range(num_opponents):
                offset = missing + 2 * i
                rival_rank = handdat[handdat[final_board + deck[offset]] + deck[offset + 1]]
                if rival_rank > hero_rank:
                    winners = 0
                    break
                if rival_rank == hero_rank:
                    winners += 1
            outcomes[winners] += 1
        pot_share = <double>outcomes[1]
        for i in range(2, num_opponents + 2):
            ties += outcomes[i]
            pot_share += <double>outcomes[i] / i

    return (int(outcomes[1]), int(ties), int(outcomes[0]), pot_share / samples)
