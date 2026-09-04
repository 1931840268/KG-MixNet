# Dataset provenance and split definitions

This directory holds everything needed to confirm that you are working with the
same images we did, and to reproduce our folds exactly. It contains metadata
only: hashes, counts, capture dates and split membership. No model outputs.

The dataset itself is CoLeaf-DB, published on Mendeley Data as
[`brfgw46wzb`](https://data.mendeley.com/datasets/brfgw46wzb); we redistribute
none of it.

## Files

| File | What it gives you |
| --- | --- |
| `archive_sha256.json` | The ten class archives our copy was built from, each with the size and SHA-256 published by Mendeley. We re-hashed all ten locally and all ten matched, so the copy behind every reported number is the published release rather than a re-encoded derivative. |
| `image_md5_manifest.json` | MD5 of each of the 1006 images, per-class counts, EXIF capture summary, and the exact-duplicate audit. Hash your own copy and compare. |
| `folds_random.json` | Validation-fold membership for the random protocol, `KFold(n_splits=5, shuffle=True, random_state=42)` over the 902 seen images. This is the protocol the main tables use. |
| `folds_group.json` | Validation-fold membership for the group-aware protocol, `GroupKFold(n_splits=5)` on capture session. |
| `capture_groups.json` | The capture-session group of each seen image, which is what `folds_group.json` is built from. |

Folds are listed by filename, not by integer index, so nothing depends on
guessing our directory traversal order.

## Numbers you can check against the paper

Running `analysis/dataset_audit.py` on a correct copy of CoLeaf-DB reproduces
every provenance figure in Section 3.2.1:

- 1006 images: 902 seen across nine directories, 104 compound target images in
  `more-deficiencies`.
- 33 groups of byte-identical images. 26 are within a single class, 22 of those
  in phosphorus, the largest class, and the remaining four in boron and
  potassium. 7 are cross-class, and all seven pair iron with magnesium or iron
  with manganese, the visually confusable pairs.
- No compound target image duplicates a seen-class image, so no zero-shot
  result is contaminated by an image the model was trained on.
- EXIF survives in 853 of the 1006 files, recording two camera bodies and
  eleven capture dates, ten of which span more than one class.

`dataset_audit.py` also carries a perceptual (dHash) screen. Near-duplicate is
a threshold you pick, not a property of the data, so we publish the screen and
not one threshold's answer: run it with `--phash-distance` to get yours.

## Reproducing the group constraint

`folds_group.json` and `capture_groups.json` together are checkable in a few
lines: map every filename in every fold to its group, and no group should
appear in two folds. The same check on `folds_random.json` finds eleven capture
dates spread across all five folds, which is what motivates the second
protocol.

```python
import json

groups = json.load(open("capture_groups.json"))["groups"]
folds = json.load(open("folds_group.json"))["validation_folds"]

seen = {}
for fold, names in folds.items():
    for n in names:
        seen.setdefault(groups[n.split("/")[1]], set()).add(fold)

print(sum(1 for f in seen.values() if len(f) > 1), "groups span more than one fold")
```

## Note on directory names

The published archive spells the potassium directory `potasium-K`. We keep the
spelling as distributed so that paths match a fresh download.
