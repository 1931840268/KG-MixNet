# -*- coding: utf-8 -*-
"""Generate the deterministic description variants used in the robustness study.

The reviewer asked for perturbations along four axes: wording, clause order,
description length and contrastive statements, plus an ablation against
independently authored descriptions. Three of the five are produced
deterministically from the original corpus by this script, so that no
favourable variant can be selected by hand:

  order   : the four numbered facets are permuted under a fixed seed
            (clause order changes, content does not)
  terse   : only the mobility statement and the first two facets are kept
            (length changes to about 56% of the original)
  nocontr : every contrastive clause is removed, nothing else is touched

The wording axis requires genuine rewriting and lives in
``variant_paraphrase.py``. The independently authored corpus is described in
the paper's data-availability statement.

Usage
-----
    python knowledge_graph/make_variants.py            # writes variants.json
    from make_variants import build                    # or call directly
"""
import io, os, re, json, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["B", "Ca", "Fe", "Healthy", "K", "Mg", "Mn", "N", "P"]

ns = {}
exec(compile(io.open(os.path.join(HERE, "class_descriptions.py"),
                     encoding="utf-8").read(), "t", "exec"), ns)
CD = ns["CLASS_DESCRIPTIONS"]

FACET = re.compile(r"(\d+)\.\s+([A-Z][A-Za-z /]+):\s*(.*?)(?=\s*\d+\.\s+[A-Z][A-Za-z /]+:|$)",
                   re.S)


def split(text):
    """Split a record into (mobility statement, [(facet name, content), ...])."""
    m = list(FACET.finditer(text))
    assert len(m) == 4, "expected 4 facets, found %d" % len(m)
    lead = text[:m[0].start()].strip()
    return lead, [(x.group(2), x.group(3).strip()) for x in m]


def join(lead, facets):
    body = " ".join("%d. %s: %s" % (i + 1, n, c) for i, (n, c) in enumerate(facets))
    return (lead + " " + body).strip()


# ---------- Clause order ----------
def v_order(text, key):
    lead, f = split(text)
    r = random.Random("order:" + key)
    idx = list(range(4))
    while True:                       # ensure the permutation differs from the original
        r.shuffle(idx)
        if idx != [0, 1, 2, 3]:
            break
    return join(lead, [f[i] for i in idx])


# ---------- Description length ----------
def v_terse(text, key):
    lead, f = split(text)
    return join(lead, f[:2])


# ---------- Contrastive statements removed ----------
CONTRAST = [
    (re.compile(r"\bUnlike\s+[^,]{2,60},\s*"), ""),                 # sentence-initial "Unlike X,"
    (re.compile(r"\bIn contrast to\s+[^,]{2,60},\s*"), ""),
    (re.compile(r",\s*(?:distinct from|unlike|as opposed to)\s+[^,.;]{2,60}"), ""),
    (re.compile(r",?\s*compared to\s+[^.;]{2,80}"), ""),
    (re.compile(r",?\s*whereas\s+[^.;]{2,90}"), ""),
    (re.compile(r",\s*which is (?:rare|absent|uncommon|not seen)[^.;]{0,70}"), ""),
]


def v_nocontr(text, key):
    out = text
    for pat, rep in CONTRAST:
        out = pat.sub(rep, out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\.\s*\.", ".", out)
    # Restore the capital if removing a clause left a lower-case sentence start
    out = re.sub(r"(?<=[.:]\s)([a-z])", lambda m: m.group(1).upper(), out)
    return out.strip()


# ---------- Independently authored corpus ----------
# Written separately, to a different template and from a different subset of the
# sources, and used as the fifth arm of the robustness study. It is not bundled
# here; see the paper's data-availability statement. The loader is therefore
# optional and its absence must not break the deterministic variants above.
def v_indep():
    path = os.path.join(HERE, "archived_graph.json")
    if not os.path.exists(path):
        return None
    g = json.load(io.open(path, encoding="utf-8"))
    d = g["descriptions"]
    return {k: d[k] for k in ORDER}


VAR = {}
for name, fn in (("order", v_order), ("terse", v_terse), ("nocontr", v_nocontr)):
    VAR[name] = {k: fn(CD[k], k) for k in ORDER}
_indep = v_indep()
if _indep is not None:
    VAR["indep"] = _indep

os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
json.dump({"original": {k: CD[k] for k in ORDER}, **VAR},
          io.open(os.path.join(HERE, "out", "variants.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("=" * 80)
print("%-10s %-10s %-12s %-12s" % ("variant", "mean chars", "length ratio", "word overlap"))
print("=" * 80)
base = {k: set(re.findall(r"[a-z]{4,}", CD[k].lower())) for k in ORDER}
print("%-10s %-10.0f %-12s %-12s" % ("original",
      sum(len(CD[k]) for k in ORDER) / 9, "1.00", "1.00"))
for name in [n for n in ("order", "terse", "nocontr", "indep") if n in VAR]:
    v = VAR[name]
    L = sum(len(v[k]) for k in ORDER) / 9
    ov = sum(len(base[k] & set(re.findall(r"[a-z]{4,}", v[k].lower()))) / max(len(base[k]), 1)
             for k in ORDER) / 9
    print("%-10s %-10.0f %-12.2f %-12.2f"
          % (name, L, L / (sum(len(CD[k]) for k in ORDER) / 9), ov))
print()
print("=== Iron record under the three deterministic perturbations ===")
for name in ("order", "terse", "nocontr"):
    print()
    print("[%s] %s" % (name, VAR[name]["Fe"]))


def build(variant, records=None):
    """Return the requested variant as {class: description}.

    ``variant`` is one of "order", "terse" or "nocontr". ``records`` defaults to
    the original corpus in ``class_descriptions.py``.
    """
    src = records if records is not None else CD
    fn = {"order": v_order, "terse": v_terse, "nocontr": v_nocontr}.get(variant)
    if fn is None:
        raise ValueError("unknown variant: %s" % variant)
    return {k: fn(src[k], k) for k in src}
