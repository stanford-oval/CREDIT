"""Two checks of the influence labels (Appendix: measurement checks), recomputed from the released outputs.

threshold    A strict rule (L=0, U=4): a piece is influential only if no valid counterfactual answer gives the suggested
             answer, and non-influential only if all of them give it or the final answer does not. For each model and
             question set: the share of the default rule's scored pieces that the strict rule keeps, and balanced
             accuracy of the untrained model and of CREDIT under both rules.
reliability  For each scored piece with c of n valid counterfactual answers giving the suggested answer, draw a rate
             from Beta(c + 1/2, n - c + 1/2), draw four new counterfactual answers, and relabel with the default rule,
             keeping the final answer fixed. Among the resampled labels that are scored, reports the share equal to the
             original label and the balanced accuracy of a predictor that reproduces the original labels (all answers,
             and answers that agree with the suggestion), averaged over draws, plus the share that stays scored.
Both use CREDIT's labels for the three main models on the test, fresh and large sets (user suggestions).
CPU only. Usage: python -m credit.label_checks [--out label_checks.json] [--draws 200]
"""
import argparse
import json
import os
import random

import credit.llm_match as lm
import credit.paper_numbers as pn
from credit.agree_conditional import RUNS

MAIN = ("Qwen3-8B", "Qwen3-14B", "DeepSeek-R1-7B")


def strict(own, reask):
    ks = [v for v in reask if v is not None]
    if len(ks) < 3 or own is None:
        return "skip"
    c = sum(ks)
    if own and c == 0:
        return True
    if (not own) or c == len(ks):
        return False
    return None


def ba_of(pairs):
    """Balanced accuracy of predictions against labels, from (prediction, label) pairs."""
    I = [p for p, l in pairs if l]; N = [p for p, l in pairs if not l]
    if not I or not N:
        return None
    return (sum(I) / len(I) + 1 - sum(N) / len(N)) / 2


def threshold(S):
    T = [dict(r, lab=strict(r["adopt"], r["re"])) for r in S]
    T = [r for r in T if r["lab"] not in ("skip", None)]
    return pn.score(S)[0], pn.score(T)[0], len(T) / len(S)


def reliability(S, draws, rng):
    unchanged, oracle, scored = {"all": [], "agree": []}, {"all": [], "agree": []}, []
    for _ in range(draws):
        new = []
        for r in S:
            ks = [v for v in r["re"] if v is not None]
            p = rng.betavariate(sum(ks) + 0.5, len(ks) - sum(ks) + 0.5)
            new.append(lm.k4(r["adopt"], [rng.random() < p for _ in range(4)]))
        scored.append(sum(n not in ("skip", None) for n in new) / len(new))
        for part, keep in (("all", lambda r: True), ("agree", lambda r: r["adopt"])):
            X = [(r["lab"], n) for r, n in zip(S, new) if keep(r) and n not in ("skip", None)]
            unchanged[part].append(sum(o == n for o, n in X) / len(X))
            oracle[part].append(ba_of(X))
    mean = lambda v: sum(v) / len(v)
    return {**{f"unchanged_{p}": mean(unchanged[p]) for p in unchanged}, **{f"oracle_ba_{p}": mean(oracle[p]) for p in oracle},
            "still_scored": mean(scored)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(pn.A, "label_checks.json"))
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng, out = random.Random(args.seed), {}
    for m in MAIN:
        for st, (root, ub, cr, sp, al) in RUNS[m].items():
            S_u, S_c = pn.rows(root, ub, sp, "syco", al), pn.rows(root, cr, sp, "syco", al)
            if not S_u or not S_c:
                print(m, st, "missing"); continue
            (bu, su, _), (bc, sc, kept) = threshold(S_u), threshold(S_c)
            rel = reliability(S_c, args.draws, rng)
            out[f"{m}|{st}"] = dict(kept=round(kept, 3), credit=[round(bc, 3), round(sc, 3)], untrained=[round(bu, 3), round(su, 3)],
                                    gain_change=round((sc - su) - (bc - bu), 3), reliability={k: round(v, 3) for k, v in rel.items()})
            print(f"{m:15s} {st:6s} strict rule keeps {kept:.0%} | CREDIT {bc:.2f} -> {sc:.2f} | untrained {bu:.2f} -> {su:.2f} | "
                  f"gain change {(sc - su) - (bc - bu):+.2f} | labels unchanged {rel['unchanged_all']:.1%} "
                  f"(agreeing {rel['unchanged_agree']:.1%}) | original labels vs resampled: BA {rel['oracle_ba_all']:.2f} "
                  f"(agreeing {rel['oracle_ba_agree']:.2f})", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
