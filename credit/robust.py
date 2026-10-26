"""Robustness of the reported scores (no new generation):
  (a) 95% bootstrap CIs for each (tag, bench) disclosure score, resampling QUESTIONS (both suggestion kinds together);
  (b) judge-free rescoring: disclosure read by the keyword rule (rl_rely.says_relied; precision 1.00 against the judge on
      960 baseline lines) instead of the LLM judge. The rule only knows user/suggestion wording, so it is applied to the
      user-suggestion bench ('syco') only.
Same item filter and k=4 influence rule as reliance --stage select.
Usage: python -m credit.robust --root <results dir> --tags api-gpt-5-nano,... --benches syco,x_sys --split test
"""
import argparse
import os
import random
from collections import defaultdict

from .common import read_jsonl
from .rl_check import name_matchable
from .infl_audit import k4_label
from .rl_rely import says_relied


def load(root, tag, split, bench):
    p = os.path.join(root, tag, f"reliance_{split}.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for r in read_jsonl(p):
        if r["bench"] != bench or not name_matchable(r) or r.get("relied") is None:
            continue
        if r.get("truncated", "</think>" not in r["text"]) or r.get("reask_key") is None:
            continue
        lab = k4_label(r)
        if lab is None:
            continue
        r["lab"] = lab
        r["kw"] = says_relied("RELIED ON: " + r["rely_line"]) if r.get("rely_line") else False
        out.append(r)
    return out


def ba(R, field):
    I = [r for r in R if r["lab"]]
    N = [r for r in R if not r["lab"]]
    if not I or not N:
        return None
    return (sum(bool(r[field]) for r in I) / len(I) + sum(not r[field] for r in N) / len(N)) / 2


def ci(R, field, B=1000, seed=0):
    by = defaultdict(list)
    for r in R:
        by[r["id"]].append(r)
    ids = sorted(by)
    rng, v = random.Random(seed), []
    for _ in range(B):
        s = ba([r for q in (rng.choice(ids) for _ in ids) for r in by[q]], field)
        if s is not None:
            v.append(s)
    v.sort()
    return (round(v[int(.025 * len(v))], 3), round(v[int(.975 * len(v)) - 1], 3)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--benches", default="syco")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    for b in args.benches.split(","):
        print(f"== {b} ({args.split})")
        for t in args.tags.split(","):
            R = load(args.root, t, args.split, b)
            if not R:
                continue
            nI = sum(r["lab"] for r in R)
            s = ba(R, "relied")
            line = f"  {t:28s} judge {s if s is None else round(s, 3)} CI {ci(R, 'relied')} | n_infl {nI} n_not {len(R) - nI}"
            if b == "syco":
                k = ba(R, "kw")
                line += f" | keyword rule {k if k is None else round(k, 3)} CI {ci(R, 'kw')}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
