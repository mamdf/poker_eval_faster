from poker_eval_faster.eval_cython.main cimport create_deck


cpdef list create_deck_wrapper(int[:] dead_cards):
    cdef int len_dead_cards = dead_cards.shape[0]
    cdef int[52] results  # Asumiendo que el máximo número de cartas es 52
    cdef int num_cards

    num_cards = create_deck(dead_cards, len_dead_cards, results)
    return [results[i] for i in range(num_cards)]
