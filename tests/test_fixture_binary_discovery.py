"""Fixture generators must locate PokerStove independently of the user's home."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize("name", ["heads_up", "three_way"])
@pytest.mark.parametrize("source", ["argument", "environment", "path", "missing"])
def test_fixture_generator_binary_discovery(tmp_path, name, source):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    binary = bin_dir / ("ps-eval.cmd" if os.name == "nt" else "ps-eval")
    binary.write_text("@exit /b 0\n" if os.name == "nt" else "#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"cases": []}))
    env = dict(os.environ, PATH=str(bin_dir) if source == "path" else "")
    env.pop("POKERSTOVE_BIN", None)
    if source == "environment":
        env["POKERSTOVE_BIN"] = str(binary)
    argv = [sys.executable, str(SCRIPTS / f"generate_pokerstove_{name}_fixture.py"),
            "--input", str(fixture)]
    if source == "argument":
        env["POKERSTOVE_BIN"] = str(tmp_path / "nonexistent")
        argv += ["--ps-eval", str(binary)]
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True)
    if source == "missing":
        assert result.returncode == 2
        assert "set POKERSTOVE_BIN or pass --ps-eval" in result.stderr
        assert json.loads(fixture.read_text()) == {"cases": []}
    else:
        assert result.returncode == 0, result.stderr
        assert "PokerStove" in json.loads(fixture.read_text())["source"]
