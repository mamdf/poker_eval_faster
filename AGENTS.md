# Repository Guidelines

## Project Structure & Module Organization
`poker_eval_faster/` contains the public Python API (`__init__.py`, `main.py`), packaged data in `data/HandRanks.dat`, and the Cython core in `eval_cython/` (`.pyx`, `.pxd`). Put functional changes in the Python API only when they do not belong in the Cython evaluator. Tests live in `tests/` and follow `test_*.py` naming. Benchmarks live in `benchmarks/`. Treat `build/` and `poker_eval_faster.egg-info/` as generated artifacts, not source.

## Build, Test, and Development Commands
Use an editable install so Cython extensions build and the CLI is registered:

```bash
python -m pip install -e .
```

Rebuild compiled extensions in place after editing `.pyx` or `.pxd` files:

```bash
python setup.py build_ext --inplace
```

Run the test suite:

```bash
python -m pytest tests -v
```

Run the lightweight benchmark script:

```bash
python benchmarks/benchmark_basic.py --iters 50000
```

Optional benchmark tests require `pytest-benchmark`:

```bash
python -m pytest benchmarks -q --benchmark-min-time=0.1
```

## Coding Style & Naming Conventions
Follow the existing Python/Cython style: 4-space indentation, `snake_case` for functions, variables, and test names, and concise docstrings only where behavior is not obvious. Keep card encodings and array dtypes consistent with the existing API (`int32` arrays, rank/suit strings like `Ac`, `Th`). Prefer small Python wrappers in `main.py`; keep performance-critical loops in `eval_cython/`.

## Testing Guidelines
Add or update pytest coverage for every behavior change, especially around duplicate-card handling, incomplete boards, rank parsing, and range notation. Place new tests in `tests/test_<feature>.py`. If a change affects speed-sensitive paths, add or update a benchmark in `benchmarks/` rather than weakening correctness tests.

## Commit & Pull Request Guidelines
Recent history uses short imperative summaries, often in Spanish, for example: `Agrega pruebas...` or `Optimiza la gestión...`. Keep commits focused and describe the main behavior or performance change. PRs should include: a short problem statement, the affected API or Cython module, exact test/benchmark commands run, and any impact on `HandRanks.dat` usage or extension rebuilding.

## Data & Configuration Notes
`poker_eval_faster/data/HandRanks.dat` is required at runtime. Do not rename or relocate it without updating packaging in `setup.py` and adding a regression test.

<!-- BEGIN AUTOGEN:consumed-by (workspace.contracts.yaml — do not edit by hand) -->
## Consumed by

If you change this repo's exposed surface, these repos may break — ⚠ marks an
edge that declares a risk. Full `what`/`risk` prose:
`uv run tooling/workspace-index/impact.py libs/poker_eval_faster`.
Refresh: `uv run tooling/workspace-index/gen_consumed_by.py --write`.

- **`libs/holdem_insights`** (import) — lazy importlib.import_module with fail-with-message fallback ⚠
- **`room-tools/history/poker-history-replay`** (uv-path) — Optional eval extra for settlement with known…
<!-- END AUTOGEN:consumed-by -->
