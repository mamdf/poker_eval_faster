# HU Lookup Artifact

`write_hu_preflop_lookup(path)` builds a preflop heads-up artifact for all `1326` canonical combos unless you pass `combo_indices=[...]` to restrict it to a subset.

## Write the full file

```python
from poker_eval_faster import write_hu_preflop_lookup

write_hu_preflop_lookup("artifacts/hu_preflop_full.bin")
```

`combo_indices` is only for tests or smoke builds:

```python
write_hu_preflop_lookup("subset.bin", combo_indices=[0, 60, 200, 500])
```

## Read and query it

```python
from poker_eval_faster import read_hu_preflop_lookup

artifact = read_hu_preflop_lookup("artifacts/hu_preflop_full.bin")
counts = artifact.lookup_combo_ids_ordered(1325, 1311)
print(counts.wins, counts.ties, counts.total)
```

Use `lookup_combo_ids_ordered(a, b)` when you want results in hero-vs-villain order. Use `lookup_combo_ids_unordered(a, b)` only if you want the stored canonical orientation.

## Full example: file -> combo strings -> win/tie/loss

For a full file:

```python
from poker_eval_faster import combo_id_to_str, read_hu_preflop_lookup

artifact = read_hu_preflop_lookup("artifacts/hu_preflop_full.bin")
hero_id = 1325
villain_id = 1311

counts = artifact.lookup_combo_ids_ordered(hero_id, villain_id)
print(combo_id_to_str(hero_id), combo_id_to_str(villain_id))
print(counts.wins, counts.ties, counts.losses)
```

For a subset file, use the artifact helpers because local indices and canonical ids may differ:

```python
from poker_eval_faster import read_hu_preflop_lookup

artifact = read_hu_preflop_lookup("subset.bin")
hero_id = 1325
villain_id = 1311

counts = artifact.lookup_combo_ids_ordered(hero_id, villain_id)
print(artifact.combo_id_str(hero_id), artifact.combo_id_str(villain_id))
print(counts.wins, counts.ties, counts.losses)
```

## How combo ids map to combos

`artifact.combos` is the combo metadata array. Each row stores:

- `combo_id`: canonical global combo id.
- `card_a`, `card_b`: the two internal card ids.
- `mask`: 52-bit dead-card mask for legality checks.

Example:

```python
artifact.combos[0]
```

may print:

```python
np.void((0, 1, 2, 3), dtype=[('combo_id', '<u2'), ('card_a', 'u1'), ('card_b', 'u1'), ('mask', '<u8')])
```

That means:

- `combo_id=0`
- `card_a=1`, `card_b=2`
- `mask=3` (`0b11`), so the first two card bits are set

For the full file, local index `i` and `combo_id` are the same. For a subset file, they are not guaranteed to match, so use `artifact.combos["combo_id"]` to recover the canonical ids.

Helpers available:

- `combo_id_to_str(combo_id)` for canonical global ids
- `artifact.combo_id_str(combo_id)` for ids present in a loaded artifact
- `artifact.local_combo_str(local_idx)` for local artifact positions

## Where win/tie counts live

The actual counts are in `artifact.entries`, not in `artifact.combos`. Each entry stores:

- `win`
- `tie`

The entries use triangular indexing over local combo positions:

```python
from poker_eval_faster import packed_pair_index

i = artifact.local_combo_index(1325)
j = artifact.local_combo_index(1311)
idx = packed_pair_index(i, j, artifact.header.combo_count)
print(artifact.entries[idx])
```

If an entry is illegal because the two combos share a card, both fields contain the sentinel value `4294967295`.

## File layout and size

The binary file is:

1. Fixed header
2. Combo metadata array
3. Triangular `win/tie` entry array

For the full `1326`-combo artifact:

- header: `40` bytes
- combo metadata: `1326 * 12 = 15,912` bytes
- triangular entries: `878,475 * 8 = 7,027,800` bytes

Total size: `7,043,752` bytes, about `6.72 MiB`.
