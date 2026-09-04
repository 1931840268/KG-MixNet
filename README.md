# KG-MixNet

**Knowledge-Guided Geometric Mixup for Zero-Shot Compound Deficiency Recognition in Coffee Leaves**

Official implementation. KG-MixNet recognises **unseen compound** nutrient
deficiencies in coffee leaves while training only on **single-deficiency** and
healthy images, as a generalized zero-shot learning problem.

The method combines a DINOv3 ViT-L/16 encoder with a lightweight residual
adapter, semantic prototypes built from a structured agronomic symptom corpus
and encoded once with MPNet, a multimodal geometric mixup branch that
synthesises visual–semantic pairs on the unit hypersphere, and a dual-constraint
objective pairing seen-class cross-entropy with cosine alignment and KL
regularisation.

---

## What is in this repository

`train_eval.py` is the pipeline that produced every number reported in the
paper. It trains one configuration over five folds, records source- and
target-domain metrics at every epoch, and writes the per-sample class scores to
disk so that the threshold analyses, bootstrap intervals and figures can be
recomputed offline without retraining.

```
train_eval.py                      training and evaluation for every configuration
knowledge_graph/
  class_descriptions.py            the nine class records used in the paper
  make_variants.py                 deterministic perturbations (order, length, contrast)
  variant_paraphrase.py            the rewritten corpus (wording axis)
  build_prototypes.py              encode any corpus into the frozen prototype bank
analysis/
  dataset_audit.py                 provenance, duplicate and contamination audit
provenance/
  archive_sha256.json              the ten source archives, with verified SHA-256
  image_md5_manifest.json          MD5 of all 1006 images, counts, duplicate audit
  folds_random.json                validation folds, random protocol
  folds_group.json                 validation folds, group-aware protocol
  capture_groups.json              capture session of each seen image
models/, utils/, scripts/          the original reference implementation
```

Every configuration in the paper is registered by name in `EXPERIMENTS` inside
`train_eval.py`, including the ablations, the mixup-geometry and level variants,
the reference models, the leave-one-nutrient-out protocol, the description
perturbations, the class-imbalance schemes and the preprocessing control.

---

## Reproducing the results

**1. Data.** Obtain CoLeaf-DB from the dataset article and place it at
`data/CoLeaf-DB`, one directory per class plus `more-deficiencies` for the
compound images. Confirm you hold the same copy:

```bash
python analysis/dataset_audit.py --root data/CoLeaf-DB
```

This prints the per-class counts, the MD5 manifest, the exact-duplicate groups
and, importantly, whether any compound target image duplicates a seen-class
image. On the copy used in the paper it does not, which is the condition the
zero-shot evaluation requires. Compare the output against
`provenance/image_md5_manifest.json` and `provenance/archive_sha256.json`, which
record our copy image by image and archive by archive; see
[`provenance/README.md`](provenance/README.md).

**2. Prototypes.** Encode the symptom corpus once:

```bash
python knowledge_graph/build_prototypes.py --variant kg
python knowledge_graph/build_prototypes.py --variant simple   # class-name ablation
```

**3. Backbone.** Download `facebook/dinov3-vitl16-pretrain-lvd1689m` to
`models/dinov3-vitl16`.

**4. Train.** Each experiment writes `runs/<name>.json` and
`runs/<name>.scores.npz`:

```bash
python train_eval.py --exp abl_full            # the reported model
python train_eval.py --exp ref_dsecn           # one of the reference models
python train_eval.py --exp lono_Fe_kg          # iron held out of training
python train_eval.py --exp P1                  # a whole group at once
```

Useful flags: `--folds`, `--epochs`, `--lr`, `--seed`, `--workers`,
`--group-file` for the group-aware split, and `--save-artifacts` for the
768-dimensional features the figures need.

**Note on `--workers`.** The value must be the same for every configuration.
The augmentation draws from the worker random stream, so changing it makes
configurations incomparable. Set `jobs × workers` to about the number of CPU
cores actually available to the process, which on a container is the cgroup
quota rather than what `nproc` reports.

---

## Notes on faithfulness

Two implementation details are worth flagging, because both are the kind that
fail silently.

**Backbone unfreezing.** The original code located the transformer blocks by
trying four attribute names and unfroze nothing if none matched, while still
printing normal parameter counts, losses and metrics. Under `transformers`
5.14.1 the DINOv3 blocks live at `model.layer` and none of those names matches,
so the same code degrades to an almost fully frozen backbone with no visible
symptom. `find_transformer_blocks` adds a version-independent fallback and the
caller asserts on the block count, so a mismatch raises rather than trains a
different model.

**Leave-one-nutrient-out.** The training loss must be restricted to the eight
seen classes. Left unrestricted, cross-entropy treats the held-out nutrient as a
negative for every sample, which trains the model never to predict it and drives
its top-1 and top-3 rates to zero. That measures a model trained not to answer,
not a model that cannot. `train_eval.py` therefore computes the training loss
over the eight seen columns only, so the held-out prototype receives no gradient
and keeps the position the knowledge graph gives it, while evaluation still
scores against all nine. This is the standard GZSL construction.

---

## Availability

The training and evaluation code, the class-description corpus and its
perturbation generators, the dataset audit, and the full provenance record —
archive hashes, the per-image MD5 manifest, the fold indices for both split
protocols and the capture-session grouping — are released here. Together they
reproduce the reported numbers from the public dataset without any further
material from us. The saved per-sample scores and the trained checkpoints are
available to the editors and reviewers on request during review, and will be
added to this repository upon publication.

## Citation

```bibtex
@article{wang2026kgmixnet,
  title   = {KG-MixNet: Knowledge-Guided Geometric Mixup for Zero-Shot
             Compound Deficiency Recognition in Coffee Leaves},
  author  = {Wang, Yuqi},
  journal = {Neural Computing and Applications},
  year    = {2026}
}
```

The dataset is CoLeaf-DB (Tuesta-Monteza, Mejia-Cabrera and Arcila-Diaz,
*Data in Brief*, 2023) and is not redistributed here.
