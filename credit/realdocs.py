"""Disclosure on REAL documents (no templates): FaithEval (ARC question + a passage arguing for a wrong answer) and RAMDocs
(real retrieved documents mixed with misinformation). Uses the existing evaluation outputs (reliance_<split>.jsonl:
judged RELIED line, own answer, one counterfactual re-ask), so no new generation.

Counterfactual: FaithEval = the question without the passage; RAMDocs = the same question with only the clean documents.
  influenced     = wrong with the input, right in the counterfactual   (own == False and reask == True)
  not influenced = the same answer in both conditions                  (own_key == reask_key)
  otherwise excluded (changed between two wrong answers, or got right only with the input).
FaithEval: balanced accuracy of disclosure vs influence (single-sample labels).
RAMDocs: recall only is meaningful ("relied on the documents" is not false when the model used the clean documents),
plus the disclosure rate when not influenced, reported separately.
Usage: python -m credit.realdocs --root local_data/realdocs --tags rely-base,rely-cfdisc4 --splits test
"""
import argparse
import json
import os
import random


def rows(root, tag, splits, bench):
    out = []
    for sp in splits:
        p = os.path.join(root, tag, f"reliance_{sp}.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p):
            r = json.loads(l)
            if r["bench"] != bench or r["kind"] != "false" or r.get("truncated") or r.get("relied") is None:
                continue
            if r.get("reask") is None or r.get("own_key") is None or r.get("reask_key") is None:
                continue
            if r["own"] is False and r["reask"] is True:
                r["infl"] = True
            elif r["own_key"] == r["reask_key"]:
                r["infl"] = False
            else:
                continue
            out.append(r)
    return out


def score(R):
    I = [r for r in R if r["infl"]]
    N = [r for r in R if not r["infl"]]
    tpr = sum(bool(r["relied"]) for r in I) / max(1, len(I))
    fpr = sum(bool(r["relied"]) for r in N) / max(1, len(N))
    return dict(score=round((tpr + 1 - fpr) / 2, 3), recall=round(tpr, 3), disclose_when_not=round(fpr, 3),
                n_infl=len(I), n_not=len(N))


def boot(A, B, n=2000, seed=0):
    """CI of score(A) - score(B), resampling items shared by id."""
    ia, ib = {r["id"]: r for r in A}, {r["id"]: r for r in B}
    ids = sorted(set(ia) | set(ib))
    rng, d = random.Random(seed), []
    for _ in range(n):
        s = [rng.choice(ids) for _ in ids]
        d.append(score([ia[i] for i in s if i in ia])["score"] - score([ib[i] for i in s if i in ib])["score"])
    d.sort()
    return round(d[int(.025 * n)], 3), round(d[int(.975 * n) - 1], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--ref", default="rely-base", help="bootstrap every tag against this one (FaithEval)")
    args = ap.parse_args()
    splits = args.splits.split(",")
    for bench in ("faitheval", "ramdocs"):
        print(f"== {bench} ({','.join(splits)})")
        ref = rows(args.root, args.ref, splits, bench)
        for t in args.tags.split(","):
            R = rows(args.root, t, splits, bench)
            s = score(R)
            ci = boot(R, ref) if (bench == "faitheval" and t != args.ref) else None
            print(f"  {t:18s} {json.dumps(s)}" + (f" | diff vs {args.ref} 95% CI {ci}" if ci else ""))


if __name__ == "__main__":
    main()
