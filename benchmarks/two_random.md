# Exact three-way versus random rivals

Run `python benchmarks/benchmark_two_random.py` in the installed environment.
It times 4 hands × 3 streets, with one initial call and 20 warm calls each.
Import/HandRanks loading and UI/process startup are outside these timings.

Local measurement (2026-09-22, Cython release build, single thread):

| Street | Warm median range across 4 hands | Largest warm call |
| --- | ---: | ---: |
| Flop | 1.18–1.70 ms | 1.86 ms |
| Turn | 0.052–0.078 ms | 0.120 ms |
| River | 0.011–0.012 ms | 0.017 ms |

These are observations, not performance guarantees. No sampling or additional
equity table is used. For each final board the kernel ranks each possible rival
hand once, then counts compatible pairs using card degrees:
`ordered_disjoint_pairs = n*(n-1) - sum(d*(d-1))`.
Apply this to losing hands for hero wins, and to losing-or-tying hands for
hero non-losses. Their difference counts top ties (not split-pot equity).
This accounts for 966,381,570 labeled-seat flop outcomes without visiting each.
