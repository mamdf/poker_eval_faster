import numpy as np
from pathlib import Path
from libc cimport stdint


path = Path(__file__).parent / '..' / 'data'


def _load_handranks_table(base_path: Path):
    data_path = base_path / 'HandRanks.dat'
    if data_path.exists():
        return np.fromfile(data_path, dtype=np.uint32)

    gzip_path = base_path / 'HandRanks.dat.gz'
    if gzip_path.exists():
        import gzip
        with gzip.open(gzip_path, 'rb') as file_obj:
            return np.frombuffer(file_obj.read(), dtype=np.uint32).copy()

    raise FileNotFoundError(
        f"Missing HandRanks table in {base_path}. Expected HandRanks.dat or HandRanks.dat.gz"
    )


dat = _load_handranks_table(path)  # eval file two plus two
# np.array to C(memory views)
cdef stdint.uint32_t[:] handdat = dat[:]



cpdef stdint.uint32_t evaluate_c(h):
    """ Takes a hand as an array of strings (as above)
    Returns a dict of the hand's stats.
    value is an integer whose value that can be compared to other hand values.
	Larger number = better hand"""
    cdef stdint.uint32_t p = 53
    cdef stdint.uint32_t suma
    cdef unsigned short i
    cdef unsigned short len_h = len(h)
    for i in range(len_h):
        suma = p + h[i]
        p = handdat[suma]

    if len_h==5 or len_h==6:
        p = handdat[p]

    return p


cdef int create_deck(int[:] dead_cards, int len_dead_cards, int results[]):
    """
    Create all deck cards (int 1 to 53) less dead cards.
    :param dead_cards: memoryview
    :param len_dead_cards: int length dead_cards
    :param results: C array will be append deck cards
    :return: len deck
    """
    cdef:
        int i, j
        int num_cards = 0
        bint flag
    for i in range(1, 53):
        flag = True
        for j in range(len_dead_cards):
            if dead_cards[j] == i:
                flag = False
                break
        if flag:
            results[num_cards] = i
            num_cards += 1

    return num_cards
