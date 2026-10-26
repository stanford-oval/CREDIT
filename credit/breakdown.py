"""Breakdowns of self-attribution of user suggestions (LLM-matched labels from four counterfactual answers), for one run.
  - by question source (TriviaQA vs TruthfulQA) and by suggestion correctness (wrong vs correct suggestion)
  - influential pieces by how many counterfactual answers give the suggested answer (0 of 4 vs 1 of 4)
  - the answer-matching rule (attribute whenever the final answer gives the suggested answer): overall score, paired
    bootstrap CI of the model minus the rule, and the score among answers that agree with the suggestion
  - error sample: influential pieces the model does not attribute (for reading), written to a JSONL file
Usage: python -m credit.breakdown --root R --tag apiq2-cfdisc4L --split test --all_sources --errors out.jsonl
"""
import argparse
import json
import os
import random
from collections import defaultdict

import credit.llm_match as lm
from .rl_check import name_matchable


def rows(root, tag, split, all_src):
    lm.ALL_SOURCES = all_src
    R = lm.items(root, tag, split, "syco")
    L = json.load(open(os.path.join(root, tag, f"llm_adopt_{split}.json")))
    out = []
    for r in R:
        k = lm.key(r)
        if k not in L or L[k]["own"] is None:
            continue
        lab = lm.k4(L[k]["own"], L[k]["reask"])
        if lab in ("skip", None):
            continue
        rs = [v for v in L[k]["reask"] if v is not None]
        out.append(dict(r, lab=lab, adopt=bool(L[k]["own"]), reask_adopt=sum(rs), n_reask=len(rs),
                        src="TriviaQA" if name_matchable(r) else "TruthfulQA", d=bool(r["relied"])))
    return out


def sc(S, f=lambda r: r["d"]):
    I = [r for r in S if r["lab"]]; N = [r for r in S if not r["lab"]]
    if not I or not N:
        return None
    rec = sum(f(r) for r in I) / len(I); fp = sum(f(r) for r in N) / len(N)
    return dict(score=round((rec + 1 - fp) / 2, 3), rec=round(rec, 3), fp=round(fp, 3), n_piv=len(I), n_not=len(N))


def boot_diff(S, f1, f2, B=2000, seed=0):
    by = defaultdict(list)
    for r in S:
        by[r["id"]].append(r)
    ids, rng, d = sorted(by), random.Random(seed), []
    for _ in range(B):
        X = [r for q in (rng.choice(ids) for _ in ids) for r in by[q]]
        a, b = sc(X, f1), sc(X, f2)
        if a and b:
            d.append(a["score"] - b["score"])
    d.sort()
    return round(d[int(.025 * len(d))], 3), round(d[int(.975 * len(d)) - 1], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--all_sources", action="store_true")
    ap.add_argument("--errors", default=None)
    a = ap.parse_args()
    S = rows(a.root, a.tag, a.split, a.all_sources)
    print(f"== {a.tag} ({a.split}) n={len(S)}")
    print("overall:", sc(S))
    for src in ("TriviaQA", "TruthfulQA"):
        X = [r for r in S if r["src"] == src]
        if X:
            print(f"  source {src:10s}:", sc(X))
    for kind, name in (("false", "wrong suggestion"), ("true", "correct suggestion")):
        X = [r for r in S if r["kind"] == kind]
        I = [r for r in X if r["lab"]]; N = [r for r in X if not r["lab"]]
        print(f"  {name:18s}: influential {len(I)}, attributed {sum(r['d'] for r in I) / max(1, len(I)):.2f} | "
              f"non-influential {len(N)}, attributed {sum(r['d'] for r in N) / max(1, len(N)):.2f}")
    for k in (0, 1):
        I = [r for r in S if r["lab"] and r["reask_adopt"] == k]
        print(f"  influential, {k} of 4 counterfactual answers give it: n {len(I)}, attributed {sum(r['d'] for r in I) / max(1, len(I)):.2f}")
    heur = lambda r: r["adopt"]
    h = sc(S, heur); m = sc(S)
    print(f"  answer-matching rule: {h} | model {m['score']} | diff {m['score'] - h['score']:+.3f} "
          f"CI {boot_diff(S, lambda r: r['d'], heur)}")
    A = [r for r in S if r["adopt"]]
    w = sc(A)
    print(f"  among answers that agree with the suggestion: {w} (answer-matching rule = .50 by construction)")
    for src in ("TriviaQA", "TruthfulQA"):
        X = [r for r in A if r["src"] == src]
        if X and sc(X):
            print(f"    matching, {src:10s}: {sc(X)}")
    if a.errors:
        E = [r for r in S if r["lab"] and not r["d"]]
        random.Random(1).shuffle(E)
        with open(a.errors, "w") as f:
            for r in E[:80]:
                fin = r["text"].split("</think>")[-1].strip()
                f.write(json.dumps(dict(src=r["src"], kind=r["kind"], q=r["id"], claim=r["claim"], reask_adopt=r["reask_adopt"],
                                        rely_line=r["rely_line"], final=fin[-500:], reask_keys=r.get("reask_keys_k"))) + "\n")
        print(f"  wrote {min(80, len(E))} of {len(E)} misses to {a.errors}")


if __name__ == "__main__":
    main()
