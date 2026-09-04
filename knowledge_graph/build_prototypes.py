#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Encode the class records into the frozen semantic prototype bank.

Each of the nine classes is one record written to the schema described in
Section 3.3 of the paper: a mobility statement followed by exactly four
numbered facets. The record is serialised to a single string and encoded once
with MPNet; the resulting 9 x 768 matrix is stored and never updated during
training, which is why the semantic branch contributes no gradient and no
inference cost.

The same script builds the ablation and perturbation banks, so every prototype
set reported in the paper comes from one code path:

    python knowledge_graph/build_prototypes.py --variant kg      # Table 7, 12
    python knowledge_graph/build_prototypes.py --variant simple  # Table 12, "Simple class names"
    python knowledge_graph/build_prototypes.py --variant order   # Table 13
    python knowledge_graph/build_prototypes.py --variant terse
    python knowledge_graph/build_prototypes.py --variant nocontr
    python knowledge_graph/build_prototypes.py --variant paraphrase
    python knowledge_graph/build_prototypes.py --variant indep

The class order is fixed at [B, Ca, Fe, Healthy, K, Mg, Mn, N, P] and must match
the label encoding used by the cross-entropy loss in ``train_eval.py``.
"""
import argparse
import os
import sys

ORDER = ["B", "Ca", "Fe", "Healthy", "K", "Mg", "Mn", "N", "P"]
HERE = os.path.dirname(os.path.abspath(__file__))


def load_records(variant):
    sys.path.insert(0, HERE)
    from class_descriptions import CLASS_DESCRIPTIONS as ORIGINAL

    if variant == "kg":
        return ORIGINAL
    if variant == "simple":
        return {c: ("a healthy coffee leaf" if c == "Healthy"
                    else "%s deficiency in a coffee leaf" % c) for c in ORDER}
    if variant == "paraphrase":
        from variant_paraphrase import PARAPHRASE
        return PARAPHRASE
    if variant in ("order", "terse", "nocontr"):
        import make_variants
        return make_variants.build(variant, ORIGINAL)
    if variant == "indep":
        path = os.path.join(HERE, "independent_corpus.py")
        if not os.path.exists(path):
            sys.exit("The independently authored corpus is not bundled here; "
                     "see the paper's data-availability statement.")
        from independent_corpus import CLASS_DESCRIPTIONS as INDEP
        return INDEP
    sys.exit("unknown variant: %s" % variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="kg",
                    choices=["kg", "simple", "order", "terse", "nocontr",
                             "paraphrase", "indep"])
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--out-dir", default="prototypes")
    args = ap.parse_args()

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("pip install torch sentence-transformers")

    records = load_records(args.variant)
    missing = [c for c in ORDER if c not in records]
    if missing:
        sys.exit("missing classes: %s" % missing)

    texts = [records[c] for c in ORDER]
    print("Encoding %d records with %s" % (len(texts), args.model))
    for c, t in zip(ORDER, texts):
        print("  %-9s %4d words" % (c, len(t.split())))

    emb = SentenceTransformer(args.model).encode(
        texts, convert_to_tensor=True, normalize_embeddings=False)
    emb = emb.float().cpu().contiguous()
    assert emb.shape == (9, 768), emb.shape

    os.makedirs(args.out_dir, exist_ok=True)
    name = ("class_embeddings.pt" if args.variant == "kg"
            else "class_embeddings_%s.pt" % args.variant)
    out = os.path.join(args.out_dir, name)
    torch.save(emb, out)
    print("Wrote %s with shape %s" % (out, tuple(emb.shape)))


if __name__ == "__main__":
    main()
