# Random-opponent equity validation with PokerStove

Compared 100,000-sample estimates across 34 scenarios and seeds 0–99:
**3,400 estimates**, with no observed difference above one percentage point.

| Reference | Scenarios / estimates | Mean absolute difference | Maximum difference |
| --- | ---: | ---: | ---: |
| PokerStove exact | 16 / 1,600 | 0.098 pp | 0.536 pp |
| PokerStove independent 10M-sample simulation | 18 / 1,800 | 0.067 pp | 0.475 pp |

The exact references cover four hands on flop/turn/river heads-up and four
three-player river cases. The simulated references cover eight players on all
postflop streets, a flush draw, a board-wide royal flush tie, ten players on
flop, and additional dead cards. The saved [snapshot](../tests/fixtures/random_equity_pokerstove_snapshot.json)
contains the reference equity, mode, sample count, timing, seed and per-case
error statistics. Results are from September 2026, PokerStove revision
`ae377e23cfd0cf2a5e2cdc11891a93307a30a65d`.

For example, `AcTc` on `9c2dAs` against seven randoms gives **42.34133%** with
100,000 samples and seed 42. PokerStove's independent 10M-sample reference gives
**42.24849%**, a difference of **0.09285 percentage points**.

These observations do not guarantee an error below 1 pp for every seed or
scenario. In particular, the observed exact-reference maximum exceeds the
95% bound of ±0.429 pp, as a probabilistic bound allows. The 10M references
are themselves estimates, with a conservative 95% radius of ±0.043 pp, so
their observed differences are not errors against an exact ground truth.
An error measured in percentage points is absolute, not relative: at very
small equity, even a small absolute error can be large in relative terms.

## Independence and regression coverage

This PokerStove checkout's `ps-eval` enumerates exactly and supports eight
known hands, but exposes no Monte Carlo option. Exact enumeration of eight
players with seven unrestricted random hands remains impractical.

The reference harness uses PokerStove's unmodified `evaluateShowdown`, its
own card encoding and evaluator, `std::mt19937_64`, and full `std::shuffle`.
It deals opponents before completing the board. The Cython implementation
uses HandRanks, PCG32, and a partial shuffle with the board dealt first.
Thus the implementations do not share their evaluator, RNG or dealing code.
Both model uniform legal deals and equal split-pot eligibility.

Snapshot regression tests use the held-out seed 777. Tolerances account for
both estimators' uncertainty with a two-sample Hoeffding bound and a union
bound over the cases; exact references contribute no sampling uncertainty.
Ordinary tests do not require PokerStove to be installed.

## Reproduce

Requires a C++ compiler, CMake, and Boost development headers/libraries.
Set `PS_SOURCE` to the local PokerStove checkout, then run from this repository:

```bash
cmake -S "$PS_SOURCE" -B build/pokerstove -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build/pokerstove --target ps-eval --parallel 4
c++ -O3 -std=c++14 -I "$PS_SOURCE/src/lib" \
  benchmarks/pokerstove_random_reference.cpp \
  build/pokerstove/lib/libpenum.a build/pokerstove/lib/libpeval.a \
  -o build/pokerstove/bin/random-reference
python benchmarks/validate_random_equity_pokerstove.py \
  --reference build/pokerstove/bin/random-reference \
  --revision "$(git -C "$PS_SOURCE" rev-parse HEAD)" \
  --output tests/fixtures/random_equity_pokerstove_snapshot.json
python -m pytest tests/test_random_equity_pokerstove_snapshot.py -v
```

The report generator allows `--samples`, `--seeds` and `--reference-samples`
to vary the comparison budget. Regeneration uses the standard library's
shuffle implementation, so different C++ standard libraries can produce
different finite-sample reference values even with the same seed.
