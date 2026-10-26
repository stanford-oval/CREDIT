"""Agreement between human annotators and the two LLM judges (reliance statements; answer matching).
Task 1 (reliance statements): annotator_*_task1.csv vs judge_check_key.json.
Task 2 (answer matching):     annotator_*_task2.csv vs match_check_key.json.
For each task, prints
per-annotator agreement with the judge (percent, Cohen's kappa), agreement by judge verdict and by source group,
pairwise inter-annotator agreement, and agreement of the human majority with the judge.
Usage: python human_eval/score.py
"""
import csv
import glob
import itertools
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = {"task1": "judge_check_key.json", "task2": "match_check_key.json"}
KEY = {}


def read(p):
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        v = r["label (yes/no)"].strip().lower()
        if v[:1] in ("y", "n"):  # yes / y / yea / no / n
            out[int(r["id"])] = v.startswith("y")
    return out


def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agree(lab, ids):
    h = [lab[i] for i in ids]; j = [KEY[i]["judge"] for i in ids]
    return sum(x == y for x, y in zip(h, j)) / len(ids), kappa(h, j), len(ids)


def report(task):
    global KEY
    KEY = {k["i"]: k for k in json.load(open(os.path.join(HERE, KEYS[task])))}
    files = sorted(glob.glob(os.path.join(HERE, f"annotator_*_{task}.csv")))
    print(f"===== {task}: {len(files)} annotator file(s)")
    A = {os.path.basename(p).rsplit("_", 1)[0]: read(p) for p in files}
    for name, lab in A.items():
        ids = sorted(set(lab) & set(KEY))
        p, k, n = agree(lab, ids)
        print(f"{name}: agreement with judge {p:.3f}, kappa {k:.3f} (n={n})")
        for v in (True, False):
            s = [i for i in ids if KEY[i]["judge"] == v]
            if s:
                print(f"   judge={'yes' if v else 'no '}: human agrees {sum(lab[i] == v for i in s) / len(s):.3f} (n={len(s)})")
        for g in sorted({KEY[i]["group"] for i in ids}):
            s = [i for i in ids if KEY[i]["group"] == g]
            print(f"   {g:18s}: {agree(lab, s)[0]:.3f} (n={len(s)})")
    for (a, la), (b, lb) in itertools.combinations(A.items(), 2):
        ids = sorted(set(la) & set(lb))
        print(f"{a} vs {b}: agreement {sum(la[i] == lb[i] for i in ids) / len(ids):.3f}, "
              f"kappa {kappa([la[i] for i in ids], [lb[i] for i in ids]):.3f} (n={len(ids)})")
    if len(A) >= 2:
        maj = {}
        for i in KEY:
            v = [lab[i] for lab in A.values() if i in lab]
            c = Counter(v)
            if v and c[True] != c[False]:
                maj[i] = c[True] > c[False]
        p, k, n = agree(maj, sorted(maj))
        print(f"human majority vs judge: agreement {p:.3f}, kappa {k:.3f} (n={n}, ties dropped)")


for t in KEYS:
    report(t)
