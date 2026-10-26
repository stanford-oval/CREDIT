"""Balanced accuracies behind the paper's main tables, recomputed from the saved outputs and the LLM-matched influence
labels (llm_adopt_<split>.json). For each (root, tag, split, bench): balanced accuracy, the share of influential and of
non-influential pieces that are attributed, counts, answer accuracy with wrong / correct suggestions, a 95% CI
(question bootstrap), and paired bootstrap differences for the listed pairs.
Usage: python -m credit.paper_numbers --out numbers.json
"""
import argparse
import json
import os
import random
from collections import defaultdict

import credit.llm_match as lm

A = os.environ.get("CREDIT_ROOT", ".")
R8, R14, RA, RD, RL = f"{A}/results", f"{A}/results_q14", f"{A}/results_api", f"{A}/results_r1q", f"{A}/results_r1l"
XB = ["x_tool", "x_peer", "x_meta", "x_sys", "x_cred"]
FRONT = ["gpt-5-nano", "gpt-5-mini", "gpt-5", "claude-haiku-4-5", "claude-sonnet-5", "gemini-3.5-flash-lite",
         "gemini-3.8-flash", "gemini-3.1-pro-preview",
         "gpt-5.5", "gpt-6.1-sol", "claude-opus-5-5", "deepseek-v4.1-flash"]  # last four: user suggestions only
SYCO8 = ["rely-base", "rely-intro", "rely-cheah", "rely-vft", "rely-vftuse", "rely-cfshuf", "rely-cfflip1", "rely-cfdisc4",
         "rely-vftL", "rely-vftuseL", "rely-vftmaskL", "rely-cfflip1L", "rely-cfshufL", "rely-cfdisc4L", "rely-cfdisc4L-s1", "rely-cfdisc4L-s2", "rely-cfdiscL", "rely-cfdisc2L"]
X8 = ["x-base", "x-cfdisc4L", "x-cfdisc4L-s1", "x-cfdisc4L-s2", "x-cfdiscL", "x-cfdisc2L", "xb-base", "xb-cfdisc4L"]


# Fresh-set (split test2) runs are named t2-<variant> / t2x-<variant> by scripts/04_evaluate.sh; the released outputs keep
# some of them under rely-<variant> / x-<variant>. Read whichever exists.
SWAP = {"rely-": "t2-", "t2-": "rely-", "x-": "t2x-", "t2x-": "x-"}


def resolve(root, tag, split):
    if split != "test2" or os.path.exists(os.path.join(root, tag, f"reliance_{split}.jsonl")):
        return tag
    for a in sorted(SWAP, key=len, reverse=True):
        if tag.startswith(a):
            alt = SWAP[a] + tag[len(a):]
            if os.path.exists(os.path.join(root, alt, f"reliance_{split}.jsonl")):
                return alt
            break
    return tag


def rows(root, tag, split, bench, all_src=False):
    lm.ALL_SOURCES = all_src
    tag = resolve(root, tag, split)
    R = lm.items(root, tag, split, bench)
    p = os.path.join(root, tag, f"llm_adopt_{split}.json")
    if not R or not os.path.exists(p):
        return []
    L = json.load(open(p))
    out = []
    for r in R:
        k = lm.key(r)
        if k not in L:
            continue
        lab = lm.k4(L[k]["own"], L[k]["reask"])
        if lab in ("skip", None):
            continue
        out.append(dict(id=r["id"], kind=r["kind"], lab=lab, d=bool(r["relied"]), own=bool(r.get("own")),
                        adopt=bool(L[k]["own"]), re=L[k]["reask"]))
        # own = answer correct; adopt = final answer gives the suggestion (LLM); re = the same for each counterfactual answer
    return out


def score(S):
    I = [r for r in S if r["lab"]]; N = [r for r in S if not r["lab"]]
    if not I or not N:
        return None
    rec = sum(r["d"] for r in I) / len(I); fp = sum(r["d"] for r in N) / len(N)
    return (rec + 1 - fp) / 2, rec, fp, len(I), len(N)


def boot(S, B=1000, seed=0):
    by = defaultdict(list)
    for r in S:
        by[r["id"]].append(r)
    ids, rng, v = sorted(by), random.Random(seed), []
    for _ in range(B):
        s = score([r for q in (rng.choice(ids) for _ in ids) for r in by[q]])
        if s:
            v.append(s[0])
    v.sort()
    return round(v[int(.025 * len(v))], 3), round(v[int(.975 * len(v)) - 1], 3)


def paired(SA, SB, B=2000, seed=0):
    ga, gb = defaultdict(list), defaultdict(list)
    for r in SA: ga[r["id"]].append(r)
    for r in SB: gb[r["id"]].append(r)
    ids, rng, d = sorted(set(ga) & set(gb)), random.Random(seed), []
    for _ in range(B):
        s = [rng.choice(ids) for _ in ids]
        a, b = score([r for q in s for r in ga[q]]), score([r for q in s for r in gb[q]])
        if a and b:
            d.append(a[0] - b[0])
    d.sort()
    full = score(SA)[0] - score(SB)[0]
    return round(full, 3), round(d[int(.025 * len(d))], 3), round(d[int(.975 * len(d)) - 1], 3)


def entry(S):
    s = score(S)
    if not s:
        return None
    acc = {k: [r["own"] for r in S if r["kind"] == k] for k in ("false", "true")}
    return dict(score=round(s[0], 5), rec=round(s[1], 5), fp=round(s[2], 5), n_infl=s[3], n_not=s[4], ci=boot(S),
                acc_wrong=round(sum(acc["false"]) / max(1, len(acc["false"])), 3),
                acc_right=round(sum(acc["true"]) / max(1, len(acc["true"])), 3))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    out, cache = {}, {}
    def get(name, root, tag, split, bench, all_src=False):
        S = rows(root, tag, split, bench, all_src); cache[name] = S
        if S and "base" in tag:  # answer-agreement heuristic: report reliance iff the final answer gives the suggestion
            H = [dict(r, d=r["adopt"]) for r in S]; cache[name + "|heur"] = H
            if entry(H): out[name + "|heur"] = entry(H)
        e = entry(S)
        if e:
            out[name] = e
            print(f"{name:48s} {e['score']:.3f} {e['ci']} rec {e['rec']:.2f} fp {e['fp']:.2f} n {e['n_infl']}/{e['n_not']} "
                  f"acc {e['acc_wrong']:.2f}/{e['acc_right']:.2f}", flush=True)
    for t in SYCO8:
        get(f"8b|test|syco|{t}", R8, t, "test", "syco")
    for t in X8:
        for b in XB:
            get(f"8b|test|{b}|{t}", R8, t, "test", b)
    for t in ["xc-base", "xc-cfdisc4L"]:
        get(f"8b|test|x_sysn|{t}", R8, t, "test", "x_sys")
    for t in ["rely-base", "rely-cheah", "rely-vft", "rely-cfdisc4", "t2-cfdisc4L", "t2-vftmaskL"]:
        get(f"8b|test2|syco|{t}", R8, t, "test2", "syco")
    for t in ["x-base", "x-cfdisc4", "t2x-cfdisc4L"]:
        for b in XB:
            get(f"8b|test2|{b}|{t}", R8, t, "test2", b)
    for t in ["apiq2-base", "apiq2-cfdisc4L"]:
        get(f"8b|apiq2|syco|{t}", R8, t, "test", "syco", True)
    get("8b|apiq2|syco|apiq2-intro", R8, "apiq2-intro", "test", "syco", True)
    for t in ["rely-base", "rely-intro", "rely-cfdisc4L"]:
        get(f"14b|test|syco|{t}", R14, t, "test", "syco")
    for t in ["rely-base", "rely-cfdisc4L"]:
        get(f"14b|test2|syco|{t}", R14, t, "test2", "syco")
    for t in ["apiq2-base", "apiq2-cfdisc4L"]:
        get(f"14b|apiq2|syco|{t}", R14, t, "test", "syco", True)
    for t in ["x-base", "x-cfdisc4L"]:
        for b in XB:
            get(f"14b|test|{b}|{t}", R14, t, "test", b)
    for t in ["rely-base", "rely-cfdisc4L"]:
        get(f"r1q|test|syco|{t}", RD, t, "test", "syco")
    for t in ["t2-base", "t2-cfdisc4L"]:
        get(f"r1q|test2|syco|{t}", RD, t, "test2", "syco")
    for t in ["apiq2-base", "apiq2-cfdisc4L"]:
        get(f"r1q|apiq2|syco|{t}", RD, t, "test", "syco", True)
    for t in ["x-base", "x-cfdisc4L"]:
        for b in XB:
            get(f"r1q|test|{b}|{t}", RD, t, "test", b)
    for t in ["rely-base", "rely-cfdisc4L"]:  # DeepSeek-R1-Distill-Llama-8B (Appendix: a Llama-based model)
        get(f"r1l|test|syco|{t}", RL, t, "test", "syco")
    for t in ["t2-base", "t2-cfdisc4L"]:
        get(f"r1l|test2|syco|{t}", RL, t, "test2", "syco")
    for t in ["apiq2-base", "apiq2-cfdisc4L"]:
        get(f"r1l|apiq2|syco|{t}", RL, t, "test", "syco", True)
    for t in ["x-base", "x-cfdisc4L"]:
        for b in XB:
            get(f"r1l|test|{b}|{t}", RL, t, "test", b)
    for m in FRONT:
        get(f"api|apiq2|syco|{m}", RA, f"api2-{m}", "test", "syco", True)
        get(f"api|105|x_sys|{m}", RA, f"api-{m}", "test", "x_sys")
        get(f"api|105|x_sysn|{m}", RA, f"api-{m}", "test", "x_sysn")
    pairs = [("8b|test|syco|rely-cfdisc4L", "8b|test|syco|rely-base"), ("8b|test|syco|rely-cfdisc4L", "8b|test|syco|rely-cheah"),
             ("8b|test|syco|rely-cfdisc4L", "8b|test|syco|rely-vft"), ("8b|test|syco|rely-cfdisc4L", "8b|test|syco|rely-cfshuf"),
             ("8b|test|syco|rely-cfdisc4L", "8b|test|syco|rely-cfflip1"), ("14b|test|syco|rely-cfdisc4L", "14b|test|syco|rely-base"),
             ("8b|test2|syco|t2-cfdisc4L", "8b|test2|syco|rely-base"), ("8b|apiq2|syco|apiq2-cfdisc4L", "8b|apiq2|syco|apiq2-base")]
    pairs += [("8b|test|syco|rely-cfdisc4L", "8b|test|syco|rely-cfflip1L"), ("8b|test|syco|rely-cfdisc4L", "8b|test|syco|rely-vftL"),
              ("8b|test|syco|rely-cfdisc4L", "8b|test|syco|rely-cfshufL"), ("8b|test|syco|rely-cfdisc4L", "8b|test|syco|rely-intro"),
              ("14b|test2|syco|rely-cfdisc4L", "14b|test2|syco|rely-base"), ("14b|apiq2|syco|apiq2-cfdisc4L", "14b|apiq2|syco|apiq2-base"),
              ("14b|test|syco|rely-cfdisc4L", "14b|test|syco|rely-intro"), ("8b|apiq2|syco|apiq2-cfdisc4L", "8b|apiq2|syco|apiq2-intro")]
    # trained model minus the answer-agreement heuristic (same questions; heuristic built from the base model's answers is
    # NOT paired, so use the trained model's own answers: report iff its own final answer gives the suggestion)
    for m, sp, t in [("8b", "test", "rely-cfdisc4L"), ("8b", "test2", "t2-cfdisc4L"), ("8b", "apiq2", "apiq2-cfdisc4L"),
                     ("14b", "test", "rely-cfdisc4L"), ("14b", "test2", "rely-cfdisc4L"), ("14b", "apiq2", "apiq2-cfdisc4L"),
                     ("r1q", "test", "rely-cfdisc4L"), ("r1q", "test2", "t2-cfdisc4L"), ("r1q", "apiq2", "apiq2-cfdisc4L"),
                     ("r1l", "test", "rely-cfdisc4L"), ("r1l", "test2", "t2-cfdisc4L"), ("r1l", "apiq2", "apiq2-cfdisc4L")]:
        a = f"{m}|{sp}|syco|{t}"
        if cache.get(a):
            cache[a + "|ownheur"] = [dict(r, d=r["adopt"]) for r in cache[a]]
            out[a + "|ownheur"] = entry(cache[a + "|ownheur"])
        pairs.append((a, a + "|ownheur"))
    pairs += [(f"8b|test|{b}|x-cfdisc4L", f"8b|test|{b}|x-base") for b in XB]
    pairs += [(f"8b|test|{b}|x-cfdiscL", f"8b|test|{b}|x-base") for b in XB]
    pairs += [(f"14b|test|{b}|x-cfdisc4L", f"14b|test|{b}|x-base") for b in XB]
    pairs += [(f"r1q|test|{b}|x-cfdisc4L", f"r1q|test|{b}|x-base") for b in XB]
    pairs += [("r1q|test|syco|rely-cfdisc4L", "r1q|test|syco|rely-base"), ("r1q|test2|syco|t2-cfdisc4L", "r1q|test2|syco|t2-base"),
              ("r1q|apiq2|syco|apiq2-cfdisc4L", "r1q|apiq2|syco|apiq2-base")]
    pairs += [(f"r1l|test|{b}|x-cfdisc4L", f"r1l|test|{b}|x-base") for b in XB]
    pairs += [("r1l|test|syco|rely-cfdisc4L", "r1l|test|syco|rely-base"), ("r1l|test2|syco|t2-cfdisc4L", "r1l|test2|syco|t2-base"),
              ("r1l|apiq2|syco|apiq2-cfdisc4L", "r1l|apiq2|syco|apiq2-base")]
    out["_pairs"] = {}
    for a, b in pairs:
        if cache.get(a) and cache.get(b):
            out["_pairs"][f"{a} - {b}"] = paired(cache[a], cache[b])
            print(f"PAIR {a} - {b}: {out['_pairs'][f'{a} - {b}']}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
