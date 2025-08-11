import os
from pathlib import Path

import pytest


@pytest.mark.order(0)
def test_handranks_dat_exists_and_nonempty():
    # Localiza el archivo sin importar el paquete (evita cargar numpy.fromfile en el import)
    project_root = Path(__file__).resolve().parents[1]
    data_file = project_root / 'poker_eval_faster' / 'data' / 'HandRanks.dat'

    assert data_file.exists(), f"No se encontró {data_file}. Sin este archivo no funciona evaluate_rank."

    size_bytes = data_file.stat().st_size
    # Umbral conservador: el archivo suele ser ~124 MB
    assert size_bytes > 120 * 1024 * 1024, (
        f"HandRanks.dat es sospechosamente pequeño ({size_bytes} bytes). Reemplázalo por el archivo correcto (~124MB)."
    )


