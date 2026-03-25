# 3-Way Class Lookup Artifact

`write_three_way_class_lookup(path)` builds a preflop `3-way` artifact over the stable `169` hand classes in `HAND_CLASSES_169`.

The full file stores one entry for each multiset of three class ids with repetition, so it has:

- `818,805` entries
- `13` exact weak-order counts per entry
- a single stored orientation per entry (`i <= j <= k`)

Use `class_indices=[...]` only for tests or smoke builds.

## Write the full file

```python
from poker_eval_faster import write_three_way_class_lookup

write_three_way_class_lookup("artifacts/three_way_preflop.bin")
```

Subset example:

```python
from poker_eval_faster import class_label_to_id, write_three_way_class_lookup

write_three_way_class_lookup(
    "subset.bin",
    class_indices=[
        class_label_to_id("AA"),
        class_label_to_id("KK"),
        class_label_to_id("QQ"),
        class_label_to_id("AKs"),
    ],
)
```

## Read and query it

```python
from poker_eval_faster import read_three_way_class_lookup

artifact = read_three_way_class_lookup("artifacts/three_way_preflop.bin")
counts = artifact.lookup_labels_ordered("AA", "KK", "QQ")
print(counts.order_counts, counts.equities)
```

Use:

- `lookup_labels_ordered(a, b, c)` or `lookup_class_ids_ordered(a, b, c)` when you want results in caller order.
- `lookup_labels_unordered(a, b, c)` only when you want the stored canonical orientation.

The stored orientation is by ascending class id, not by poker strength.

## Stable 169-class ordering

`HAND_CLASSES_169` is the canonical class list used by the file. The order is:

1. pairs: `AA .. 22`
2. suited non-pairs: `AKs .. 32s`
3. offsuit non-pairs: `AKo .. 32o`

Helpers:

- `class_label_to_id("AKs") -> int`
- `class_id_to_label(13) -> str`
- `class_combo_ids("AKo") -> tuple[int, ...]`

## How entries are indexed

The artifact stores exactly one row for each non-decreasing triple of local class positions:

```text
0 <= i <= j <= k < class_count
```

The row index follows the same order as:

```python
for i in range(class_count):
    for j in range(i, class_count):
        for k in range(j, class_count):
            yield i, j, k
```

The packed index is:

```python
from math import comb

prefix_i = comb(n + 2, 3) - comb(n - i + 2, 3)
prefix_j = comb(n - i + 1, 2) - comb(n - j + 1, 2)
idx = prefix_i + prefix_j + (k - j)
```

where `n = class_count`.

The helper exposed by Python is:

```python
packed_class_triple_index(i, j, k, n)
```

and the inverse helper is:

```python
unpack_class_triple_index(idx, n)
```

## How ordered lookup works

Suppose the caller asks for:

```text
AA, QQ, KK
```

The artifact does not store that exact order separately. It stores the row for:

```text
AA, KK, QQ
```

because the stored key is sorted by class id.

The row contains the `13` counts in the stored player order:

- `A>B>C`
- `A>C>B`
- `B>A>C`
- `B>C>A`
- `C>A>B`
- `C>B>A`
- `A=B>C`
- `A=C>B`
- `B=C>A`
- `A>B=C`
- `B>A=C`
- `C>A=B`
- `A=B=C`

`lookup_*_ordered(...)` applies a fixed permutation over those `13` buckets so they match the original caller order again. Rust can do exactly the same thing:

1. map labels to class ids
2. sort the three ids and remember which original player each sorted position came from
3. read the stored row
4. permute the `13` buckets back to caller order

The six possible stored-to-original player mappings are:

- `(0, 1, 2)`
- `(0, 2, 1)`
- `(1, 0, 2)`
- `(1, 2, 0)`
- `(2, 0, 1)`
- `(2, 1, 0)`

Those correspond to the six permutations of `A/B/C`.

## File layout

The binary file is little-endian and has three sections:

1. fixed header
2. class metadata array
3. flat `entry_count x 13 x uint64` array

Header fields:

- `magic`: `PTW1`
- `version`
- `universe_class_count`
- `class_count`
- `entry_count`
- `legal_count`
- `order_count`
- `total_runouts`
- `header_size`
- `class_size`
- `entry_size`

Class metadata record:

- `class_id: uint16`
- `label[4]: ASCII`
- `combo_count: uint8`
- `flags: uint8` reserved

Entry record:

- `13 x uint64`

There is no sentinel. Impossible class triples, for example `AA,AA,AA`, are stored as thirteen zeros, so `sum(counts) == 0`.

## File size

For the full `169`-class artifact:

- header: `44` bytes
- class metadata: `169 * 8 = 1,352` bytes
- entries: `818,805 * 104 = 85,155,720` bytes

Total size: `85,157,116` bytes, about `81.21 MiB`.

## Rust navigation sketch

At read time Rust only needs:

1. read the header
2. read `class_count` class records
3. read the flat `u64` entry matrix
4. compute the packed row index
5. if the query order differs from sorted class-id order, permute the 13 buckets

No Python-specific object model is required. The file is designed to be readable with plain little-endian byte parsing or with zero-copy structs.
