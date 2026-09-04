#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provenance and duplicate audit for the CoLeaf-DB working copy.

Reported in Section 3.2.1 of the paper. The audit answers three questions and
writes a manifest that lets a reader confirm they hold the same data:

  1. Provenance   - MD5 of every image, per-class counts, and the capture
                    metadata (camera body, capture date) that survives in EXIF.
  2. Duplicates   - exact byte-level duplicate groups, split into within-class
                    (a training/validation leakage risk) and cross-class (the
                    same photograph filed under two labels in the published
                    archive).
  3. Contamination - whether any compound target image duplicates a seen-class
                    image, which is the condition that would invalidate the
                    zero-shot evaluation.

A perceptual-hash screen is included with a user-settable threshold, since
"near-duplicate" has no canonical definition; the paper reports the exact-hash
result and states that the perceptual threshold is the reader's to choose.

Usage
-----
    python analysis/dataset_audit.py --root data/CoLeaf-DB [--phash-distance 5]

The directory layout expected is the one distributed with CoLeaf-DB: one
directory per class, plus ``more-deficiencies`` holding the compound images.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required: pip install Pillow")

IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
COMPOUND_DIR = "more-deficiencies"


def walk_images(root):
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if os.path.splitext(name)[1] in IMG_EXT:
                yield cls, name, os.path.join(d, name)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dhash(path, size=8):
    """64-bit difference hash; robust to re-encoding, sensitive to content."""
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = list(g.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | int(row[c] < row[c + 1])
    return bits


def exif_of(path):
    try:
        with Image.open(path) as im:
            raw = im._getexif() or {}
        return {TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CoLeaf-DB root directory")
    ap.add_argument("--phash-distance", type=int, default=5,
                    help="Hamming distance below which two dHashes count as near-duplicate")
    ap.add_argument("--out", default="dataset_audit.json")
    args = ap.parse_args()

    files = list(walk_images(args.root))
    if not files:
        sys.exit("No images found under %s" % args.root)
    print("Images found: %d" % len(files))

    counts = collections.Counter(c for c, _, _ in files)
    print("\nPer-class counts")
    for c, n in sorted(counts.items()):
        print("  %-22s %4d" % (c, n))

    # ---------------------------------------------------------------- MD5
    by_md5 = collections.defaultdict(list)
    manifest = {}
    for cls, name, path in files:
        h = md5(path)
        by_md5[h].append((cls, name))
        manifest["%s/%s" % (cls, name)] = h
    dup = {h: v for h, v in by_md5.items() if len(v) > 1}

    within = {h: v for h, v in dup.items() if len({c for c, _ in v}) == 1}
    cross = {h: v for h, v in dup.items() if len({c for c, _ in v}) > 1}
    print("\nExact duplicate groups: %d  (within-class %d, cross-class %d)"
          % (len(dup), len(within), len(cross)))
    for h, v in sorted(cross.items()):
        print("  cross-class: %s" % ", ".join("%s/%s" % x for x in v))

    contaminating = [v for v in dup.values()
                     if any(c == COMPOUND_DIR for c, _ in v) and any(c != COMPOUND_DIR for c, _ in v)]
    print("\nTarget-to-seen duplicate groups: %d" % len(contaminating))
    if contaminating:
        for v in contaminating:
            print("  !! %s" % ", ".join("%s/%s" % x for x in v))
    else:
        print("  none - no compound target image duplicates a seen-class image")

    # ---------------------------------------------------------------- EXIF
    models, dates = collections.Counter(), collections.Counter()
    with_exif = 0
    for cls, name, path in files:
        e = exif_of(path)
        if e:
            with_exif += 1
        if e.get("Model"):
            models["%s %s" % (e.get("Make", "").strip(), str(e["Model"]).strip())] += 1
        if e.get("DateTime"):
            dates[str(e["DateTime"]).split()[0]] += 1
    print("\nEXIF present in %d of %d files" % (with_exif, len(files)))
    print("  camera bodies: %d" % len(models))
    for m, n in models.most_common():
        print("    %-38s %4d" % (m, n))
    print("  capture dates: %d" % len(dates))

    # ---------------------------------------------------------------- dHash
    print("\nPerceptual screen at Hamming distance <= %d" % args.phash_distance)
    hashes = [(cls, name, dhash(path)) for cls, name, path in files]
    near = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if bin(hashes[i][2] ^ hashes[j][2]).count("1") <= args.phash_distance:
                near += 1
    print("  near-duplicate pairs: %d" % near)
    print("  (threshold is a choice, not a property of the data; the paper reports"
          "\n   the exact-hash result above and leaves this screen to the reader)")

    json.dump({"n_images": len(files), "per_class": dict(counts), "md5": manifest,
               "exact_duplicate_groups": {h: v for h, v in dup.items()},
               "target_to_seen_duplicates": contaminating,
               "camera_bodies": dict(models), "capture_dates": dict(dates),
               "near_duplicate_pairs": near, "phash_distance": args.phash_distance},
              open(args.out, "w"), indent=1)
    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
