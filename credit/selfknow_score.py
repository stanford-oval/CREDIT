"""Asking the model whether it knows (Appendix: Asking the Model Whether It Knows, Table: selfknow). Compares CREDIT
with P(IK) (credit.pik) among answers that agree with the suggestion; P(IK) uses the threshold that maximizes balanced
accuracy on each evaluation set.
Usage: python -m credit.selfknow_score DIR [OUT]   (DIR holds pik_<model>.json; OUT defaults to $CREDIT_ROOT/selfknow.json)
"""
import json, os, random, sys
from collections import defaultdict
import credit.paper_numbers as pn
from credit.agree_conditional import RUNS
S = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get("CREDIT_ROOT", "."), "selfknow.json")
KEYM = {"Qwen3-8B": "qwen3-8b", "Qwen3-14B": "qwen3-14b", "DeepSeek-R1-7B": "r1-qwen-7b"}
def ba(rows, pred):
    I = [r for r in rows if r["lab"]]; N = [r for r in rows if not r["lab"]]
    if not I or not N: return None
    return (sum(pred(r) for r in I) / len(I) + 1 - sum(pred(r) for r in N) / len(N)) / 2
def auroc(rows, sc):  # Mann-Whitney with average ranks for ties
    xs = sorted((sc(r), r["lab"]) for r in rows); rk = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j < len(xs) and xs[j][0] == xs[i][0]: j += 1
        for k in range(i, j): rk[k] = (i + j + 1) / 2
        i = j
    P = sum(l for _, l in xs); N = len(xs) - P
    return (sum(rk[k] for k in range(len(xs)) if xs[k][1]) - P * (P + 1) / 2) / (P * N)
def best(rows, sc):  # max over thresholds t of BA(score >= t), one sorted sweep
    xs = sorted(((sc(r), r["lab"]) for r in rows), reverse=True)
    P = sum(l for _, l in xs); N = len(xs) - P
    if not P or not N: return 0
    tp = fp = 0; b = 0; i = 0
    while i < len(xs):
        v = xs[i][0]
        while i < len(xs) and xs[i][0] == v:
            tp += xs[i][1]; fp += not xs[i][1]; i += 1
        b = max(b, (tp / P + 1 - fp / N) / 2)
    return b
def boot(rows, sc, B=300):
    by = defaultdict(list)
    for r in rows: by[r["id"]].append(r)
    ids = sorted(by); rng = random.Random(0); D = []
    for _ in range(B):
        s = [x for q in (rng.choice(ids) for _ in ids) for x in by[q]]
        c = ba(s, lambda r: r["d"])
        if c is not None: D.append(c - best(s, sc))
    D.sort(); return D[int(.025 * len(D))], D[int(.975 * len(D)) - 1]
out = {}
for m, sets in RUNS.items():
    km = KEYM[m]
    pik = json.load(open(f"{S}/pik_{km}.json")) if os.path.exists(f"{S}/pik_{km}.json") else None
    for st, (root, ub, cr, sp, al) in sets.items():
        R = [r for r in pn.rows(root, cr, sp, "syco", al) if r["adopt"]]
        rec = {"credit": round(ba(R, lambda r: r["d"]), 3), "n": [sum(r["lab"] for r in R), sum(not r["lab"] for r in R)]}
        line = f"{m:15s} {st:9s} CREDIT {rec['credit']:.2f}"
        if pik:
            Rp = [r for r in R if pik.get(r["id"]) is not None]; sc = lambda r: 1 - pik[r["id"]]
            b = best(Rp, sc); ci = boot(Rp, sc); rec["pik"] = dict(ba=round(b, 3), auroc=round(auroc(Rp, sc), 3), diff=round(ba(Rp, lambda r: r["d"]) - b, 3), ci=ci)
            line += f" | P(IK) {b:.2f} (AUROC {rec['pik']['auroc']:.2f}; diff {rec['pik']['diff']:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}])"
        out[f"{m}|{st}"] = rec; print(line, flush=True)
json.dump(out, open(OUT, "w"), indent=1)

