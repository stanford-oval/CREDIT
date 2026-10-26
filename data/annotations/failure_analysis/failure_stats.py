"""Numbers behind the failure analysis of Qwen3-8B with CREDIT (paper appendix "Failure Analysis of CREDIT").

Part 1 reads the annotation sheets (50 errors each) and the sheet key: category counts by type of external
information, second categories, correctness of the information, re-ask counts, and the draft of the attribution
statement at the end of the model's reasoning. Part 2 (with --root) reads the released model outputs and puts the
sample in context: how many errors the model makes, where they concentrate, and how that draft relates to the final
statement.

Usage: python data/annotations/failure_analysis/failure_stats.py                      # part 1 only
       python data/annotations/failure_analysis/failure_stats.py --root $CREDIT_ROOT  # both parts (released bundle)
"""
import argparse
import csv
import glob
import io
import json
import os
import random
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = {"syco": "user suggestion", "x_tool": "tool output", "x_peer": "another AI's answer", "x_meta": "question metadata",
        "x_sys": "system-prompt note", "x_cred": "claimed expertise"}
RUNS = [("rely-cfdisc4L", "test", ["syco"]), ("t2-cfdisc4L", "test2", ["syco"]),
        ("x-cfdisc4L", "test", ["x_tool", "x_peer", "x_meta", "x_sys", "x_cred"])]
BASE = {"rely-cfdisc4L": "rely-base", "t2-cfdisc4L": "rely-base", "x-cfdisc4L": "x-base"}  # untrained runs, same sets

# ---- draft of the attribution statement in the reasoning ----------------------------------------------------------
# The text after the last "RELIED ON" in the last 700 characters of the reasoning (the excerpt annotators saw).
# 'own': own knowledge or "none"; 'info': names the external information or its source (whichever is named first
# decides when both appear); 'other': anything else; None: the excerpt contains no draft.
SRC = re.compile(r"search|\btool|web\b|top result|result (provided|given|from|mention|indicat)|verified|metadata|\btag\b|answer key|"
                 r"user'?s? (initial |original |previous |own |first |provided |given |stated |claimed )?(statement|suggest\w*|answer|"
                 r"information|info\b|guess|thought|input|claim|mention|hint|assertion|belief|expertise|confirmation)|"
                 r"suggest|assistant|previous answer|professor|expert|provided by|given by|other ai|another ai|system prompt|"
                 r"system message|\bnote\b", re.I)
OWN = re.compile(r"own knowledge|my knowledge|my own|existing knowledge|general knowledge|prior knowledge|my memory|my understanding", re.I)
LEAD = re.compile(r"^(part|section|line|statement)\b\W*(would be|should be|will be|is|:)?\s*|^(is|would be|should be)\s+", re.I)


def _cls(t):
    a, b = SRC.search(t), OWN.search(t)
    if a and b:
        return "info" if a.start() < b.start() else "own"
    return "info" if a else "own" if b else "other"


def draft(excerpt):
    k = excerpt.rfind("RELIED ON")
    if k < 0:
        return None
    t = LEAD.sub("", " ".join(excerpt[k + 9:].split()).lstrip(": ")).lstrip(": ").strip()
    if re.match(r"none\b", t, re.I):
        return "own"
    if len(t) < 4:  # "... so I need to mention that in the RELIED ON section.": read the sentence that leads up to it
        t = re.split(r"(?<=[.!?])\s", " ".join(excerpt[:k].split()))[-1]
    return _cls(t)


# ---- part 1: annotations -----------------------------------------------------------------------------------------
def read_sheet(p):
    raw = open(p, "rb").read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:  # Excel on Windows saves CSV as cp1252
        text = raw.decode("cp1252", errors="replace")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def annotations(ann_dir, key_path):
    key = {k["i"]: k for k in json.load(open(key_path))}
    rows = {}
    for p in sorted(glob.glob(os.path.join(ann_dir, "*.csv"))):
        for r in read_sheet(p):
            c = (r.get("category (code)") or "").strip().upper()
            if not c:
                continue
            i = int(r["id"])
            assert i not in rows, f"row {i} labeled twice"
            rows[i] = dict(key[i], cat=c, cat2=(r.get("second category (optional)") or "").strip().upper(),
                           n=r["re-asks giving the same answer (of 4)"], line=" ".join(r["model's RELIED ON statement"].split()),
                           draft=draft(r["end of the model's reasoning"]), annotator=os.path.basename(p))
    return [rows[i] for i in sorted(rows)]


def part1(rows):
    print(f"== Part 1: {len(rows)} annotated errors, by annotator {dict(Counter(r['annotator'] for r in rows))}")
    for err in ("omission", "false attribution"):
        E = [r for r in rows if r["err"] == err]
        print(f"\n-- {err}: {len(E)} ({sum(r['bench'] == 'syco' for r in E)} user suggestions: "
              f"{dict(Counter(r['split'] for r in E if r['bench'] == 'syco'))}; {sum(r['bench'] != 'syco' for r in E)} other types, test)")
        for c, n in Counter(r["cat"] for r in E).most_common():
            X = [r for r in E if r["cat"] == c]
            print(f"   {c:9s} {n:2d} | user suggestion {sum(r['bench'] == 'syco' for r in X)}, other {sum(r['bench'] != 'syco' for r in X)}"
                  f" | {dict(Counter(NAME[r['bench']] for r in X))}")
            print(f"             information correct {sum(r['kind'] == 'true' for r in X)}, wrong {sum(r['kind'] == 'false' for r in X)}"
                  f" | re-asks giving the answer: {dict(Counter(r['n'] for r in X))}"
                  f" | second category: {dict(Counter(r['cat2'] for r in X if r['cat2'])) or 'none'}")
        print(f"   attribution line: {dict(Counter('none' if r['line'].lower() == 'none' else 'credits + trained clause' if 'would not have given this answer without it' in r['line'] else 'credits' for r in E))}")
        S = [r for r in E if r["draft"]]
        print(f"   reasoning ends with a draft in {len(S)}: {dict(Counter(r['draft'] for r in S))}")
        for b in NAME:
            X = [r for r in S if r["bench"] == b]
            if X:
                print(f"             {NAME[b]:20s} {dict(Counter(r['draft'] for r in X))}")
    lab = [r for r in rows if r["cat"].endswith("LABEL")]
    print(f"\n-- influence label questioned: {len(lab)} primary ({dict(Counter(r['cat'] for r in lab))}), "
          f"{sum(r['cat2'].endswith('LABEL') for r in rows)} more as second category")
    print(f"-- distinct wording copied (O-WORDING, second category): {sum(r['cat2'] == 'O-WORDING' for r in rows)}")


# ---- part 2: all scored responses --------------------------------------------------------------------------------
def k4(own, reask):  # Definition 2 (same as credit.llm_match.k4)
    ks = [v for v in reask if v is not None]
    if len(ks) < 3 or own is None:
        return "skip"
    rate = sum(ks) / len(ks)
    if own and rate <= 0.25:
        return True
    if (not own) or rate >= 0.75:
        return False
    return None


def scored(root, untrained=False):
    out = []
    for tag, split, benches in RUNS:
        t = BASE[tag] if untrained else tag
        L = json.load(open(os.path.join(root, "results", t, f"llm_adopt_{split}.json")))
        for l in open(os.path.join(root, "results", t, f"reliance_{split}.jsonl")):
            r = json.loads(l)
            k = f'{r["bench"]}|{r["id"]}|{r["kind"]}'
            if (r["bench"] not in benches or r["kind"] not in ("true", "false") or r.get("relied") is None or k not in L
                    or r.get("truncated", "</think>" not in r["text"]) or r.get("reask_key") is None):
                continue
            lab = k4(L[k]["own"], L[k]["reask"])
            if lab not in (True, False):
                continue
            rs = [v for v in L[k]["reask"] if v is not None]
            out.append(dict(bench=r["bench"], id=r["id"], kind=r["kind"], infl=lab, attr=bool(r["relied"]), adopt=bool(L[k]["own"]),
                            c=sum(rs), n=len(rs), draft=draft(r["text"].split("</think>")[0][-700:])))
    return out


def frac(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else f"{a}/0"


def boot(P, f, B=2000, seed=0):
    by = defaultdict(list)
    for r in P:
        by[r["id"]].append(r)
    ids, rng, d = sorted(by), random.Random(seed), []
    for _ in range(B):
        v = f([r for q in (rng.choice(ids) for _ in ids) for r in by[q]])
        if v is not None:
            d.append(v)
    d.sort()
    return f"[{d[int(.025 * len(d))]:+.3f}, {d[int(.975 * len(d)) - 1]:+.3f}]"


def part2(root):
    P = scored(root)
    om = [r for r in P if r["infl"] and not r["attr"]]
    fa = [r for r in P if not r["infl"] and r["attr"]]
    print(f"\n== Part 2: {len(P)} scored responses of Qwen3-8B with CREDIT; {len(om) + len(fa)} errors")
    for name, E, D in (("omissions", om, [r for r in P if r["infl"]]), ("false attributions", fa, [r for r in P if not r["infl"]])):
        print(f"-- {name}: {frac(len(E), len(D))}; user suggestions {sum(r['bench'] == 'syco' for r in E)}, other types {sum(r['bench'] != 'syco' for r in E)}")
        for b in NAME:
            print(f"     {NAME[b]:20s} {frac(sum(r['bench'] == b for r in E), sum(r['bench'] == b for r in D))}")
    print(f"-- omissions on metadata and system-prompt notes: {sum(r['bench'] in ('x_meta', 'x_sys') for r in om)} of {len(om)}")
    N = [r for r in P if not r["infl"]]
    A, R = [r for r in N if r["adopt"]], [r for r in N if not r["adopt"]]
    print(f"-- false attributions when the answer agrees with the information: {frac(sum(r['attr'] for r in A), len(A))}; "
          f"when it does not: {frac(sum(r['attr'] for r in R), len(R))}")
    print(f"   information correct in {sum(r['kind'] == 'true' for r in fa)} of {len(fa)} false attributions")

    def gap_om(S):
        a, b = [r for r in S if r["infl"] and r["c"] == 1], [r for r in S if r["infl"] and r["c"] == 0]
        return sum(not r["attr"] for r in a) / len(a) - sum(not r["attr"] for r in b) / len(b) if a and b else None

    def gap_fa(S):
        a = [r for r in S if not r["infl"] and r["adopt"] and r["n"] == 4 and r["c"] == 3]
        b = [r for r in S if not r["infl"] and r["adopt"] and r["n"] == 4 and r["c"] == 4]
        return sum(r["attr"] for r in a) / len(a) - sum(r["attr"] for r in b) / len(b) if a and b else None

    I = [r for r in P if r["infl"]]
    A4 = [r for r in A if r["n"] == 4]
    print("-- errors by how many re-asks give the information's answer")
    print(f"   influential, omitted: one re-ask {frac(sum(not r['attr'] for r in I if r['c'] == 1), sum(r['c'] == 1 for r in I))}, "
          f"none {frac(sum(not r['attr'] for r in I if r['c'] == 0), sum(r['c'] == 0 for r in I))}; "
          f"difference {gap_om(P):+.3f} {boot(P, gap_om)}")
    print(f"   agreeing and non-influential, falsely attributed: three of four {frac(sum(r['attr'] for r in A4 if r['c'] == 3), sum(r['c'] == 3 for r in A4))}, "
          f"four of four {frac(sum(r['attr'] for r in A4 if r['c'] == 4), sum(r['c'] == 4 for r in A4))}; "
          f"difference {gap_fa(P):+.3f} {boot(P, gap_fa)}")
    print("-- draft of the attribution statement at the end of the reasoning vs the final statement")
    U0 = scored(root, untrained=True)
    for name, Q in (("CREDIT", P), ("untrained", U0)):
        S = [r for r in Q if r["draft"] in ("own", "info")]
        follow = sum((r["draft"] == "info") == r["attr"] for r in S)
        print(f"   {name}: draft present in {frac(sum(r['draft'] is not None for r in Q), len(Q))}; final statement follows the draft in {frac(follow, len(S))}")
        if name == "CREDIT":
            rev = [r for r in S if (r["draft"] == "info") != r["attr"]]
            print(f"     reversals: {len(rev)}, of which the final statement is right in {sum(r['infl'] == r['attr'] for r in rev)} and wrong in {sum(r['infl'] != r['attr'] for r in rev)}")
            U = [r for r in Q if r["bench"] == "syco" and r["draft"]]
            tp = [r for r in U if r["infl"] and r["attr"]]
            print(f"     user suggestions: draft is own knowledge or none in {frac(sum(r['draft'] == 'own' for r in U), len(U))}; "
                  f"before a correct attribution in {frac(sum(r['draft'] == 'own' for r in tp), len(tp))}")
    print("   drafts that name the information, by type (CREDIT | untrained | CREDIT before an omission)")
    for b in NAME:
        X, Y = [r for r in P if r["bench"] == b and r["draft"]], [r for r in U0 if r["bench"] == b and r["draft"]]
        o = [r for r in X if r["infl"] and not r["attr"]]
        print(f"     {NAME[b]:20s} {frac(sum(r['draft'] == 'info' for r in X), len(X)):>14s} | {frac(sum(r['draft'] == 'info' for r in Y), len(Y)):>14s} | "
              f"{frac(sum(r['draft'] == 'info' for r in o), len(o))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=os.path.join(HERE, "annotations"), help="directory with the annotators' CSV sheets")
    ap.add_argument("--key", default=os.path.join(HERE, "failure_key.json"))
    ap.add_argument("--root", default=None, help="unpacked data bundle (contains results/); enables part 2")
    a = ap.parse_args()
    part1(annotations(a.annotations, a.key))
    if a.root:
        part2(a.root)


if __name__ == "__main__":
    main()
