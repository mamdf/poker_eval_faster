import os
from pathlib import Path

import pytest


@pytest.mark.order(0)
def test_handranks_dat_exists_and_nonempty():
    # Localiza el archivo sin importar el paquete (evita cargar numpy.fromfile en el import)
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'poker_eval_faster' / 'data'
    data_file = data_dir / 'HandRanks.dat'
    gzip_file = data_dir / 'HandRanks.dat.gz'

    assert data_file.exists() or gzip_file.exists(), (
        f"No se encontró {data_file} ni {gzip_file}. Sin ese archivo no funciona evaluate_rank."
    )

    if data_file.exists():
        size_bytes = data_file.stat().st_size
        # Umbral conservador: el archivo suele ser ~124 MB
        assert size_bytes > 120 * 1024 * 1024, (
            f"HandRanks.dat es sospechosamente pequeño ({size_bytes} bytes). Reemplázalo por el archivo correcto (~124MB)."
        )
    else:
        size_bytes = gzip_file.stat().st_size
        assert size_bytes > 20 * 1024 * 1024, (
            f"HandRanks.dat.gz es sospechosamente pequeño ({size_bytes} bytes). Reemplázalo por el archivo correcto (~29MB)."
        )

