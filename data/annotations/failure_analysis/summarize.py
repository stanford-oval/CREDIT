"""Summarize failure-analysis annotations: category counts by error type and by type of external information, label
noise rate, and agreement between annotators on the primary category.
Usage: python failure_analysis/summarize.py
"""
import csv
import glob
import itertools
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = {k["i"]: k for k in json.load(open(os.path.join(HERE, "failure_key.json")))}
COL = "category (code)"


def read(p):
    out = {}
    raw = open(p, "rb").read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:  # Excel on Windows saves CSV as cp1252
        text = raw.decode("cp1252", errors="replace")
    for r in csv.DictReader(text.splitlines()):
        c = (r.get(COL) or "").strip().upper()
        if c:
            out[int(r["id"])] = c
    return out


A = {os.path.splitext(os.path.basename(p))[0]: read(p) for p in sorted(glob.glob(os.path.join(HERE, "annotations", "*.csv")))}
for name, lab in A.items():
    print(f"===== {name}: {len(lab)} labeled rows")
    for err in ("omission", "false attribution"):
        ids = [i for i in lab if KEY[i]["err"] == err]
        c = Counter(lab[i] for i in ids)
        print(f"  {err} (n={len(ids)}): " + ", ".join(f"{k} {v} ({v / len(ids):.0%})" for k, v in c.most_common()))
    by = defaultdict(Counter)
    for i, c in lab.items():
        by["user suggestion" if KEY[i]["bench"] == "syco" else KEY[i]["bench"]][c] += 1
    for b, c in sorted(by.items()):
        print(f"  {b:16s}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    noise = sum(c.endswith("LABEL") for c in lab.values())
    print(f"  questionable influence labels: {noise}/{len(lab)} ({noise / max(1, len(lab)):.0%})")
for (a, la), (b, lb) in itertools.combinations(A.items(), 2):
    ids = sorted(set(la) & set(lb))
    if ids:
        print(f"{a} vs {b}: same primary category on {sum(la[i] == lb[i] for i in ids)}/{len(ids)}")
