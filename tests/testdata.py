from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_cases(filename: str) -> list[dict[str, Any]]:
    payload = json.loads((FIXTURES_DIR / filename).read_text())
    return payload["cases"]
