"""Self-attribution among answers that agree with the suggestion (Table: agreement), where the answer-agreement rule
("attribute whenever the final answer gives the suggested answer") is at chance by construction.
For each model and question set: balanced accuracy among agreeing answers, 95% question-bootstrap CI, and the number of
influential / non-influential agreeing cases, for the untrained model and CREDIT.
Usage: python -m credit.agree_conditional [--out agree_conditional.json]
"""
import argparse
import json
import os

import credit.paper_numbers as pn

A = os.environ.get("CREDIT_ROOT", ".")
R8, R14, RD, RL = f"{A}/results", f"{A}/results_q14", f"{A}/results_r1q", f"{A}/results_r1l"
RUNS = {  # model: {set: (root, untrained tag, CREDIT tag, split, include TruthfulQA)}
    "Qwen3-8B": {"test": (R8, "rely-base", "rely-cfdisc4L", "test", False), "fresh": (R8, "rely-base", "t2-cfdisc4L", "test2", False),
                 "large": (R8, "apiq2-base", "apiq2-cfdisc4L", "test", True)},
    "Qwen3-14B": {"test": (R14, "rely-base", "rely-cfdisc4L", "test", False), "fresh": (R14, "rely-base", "rely-cfdisc4L", "test2", False),
                  "large": (R14, "apiq2-base", "apiq2-cfdisc4L", "test", True)},
    "DeepSeek-R1-7B": {"test": (RD, "rely-base", "rely-cfdisc4L", "test", False), "fresh": (RD, "t2-base", "t2-cfdisc4L", "test2", False),
                       "large": (RD, "apiq2-base", "apiq2-cfdisc4L", "test", True)},
}
# DeepSeek-R1-Distill-Llama-8B (Appendix: a Llama-based model), kept apart because the self-knowledge controls and the
# label checks cover only the three main models
LLAMA = {"DeepSeek-R1-Llama-8B": {"test": (RL, "rely-base", "rely-cfdisc4L", "test", False), "fresh": (RL, "t2-base", "t2-cfdisc4L", "test2", False),
                                  "large": (RL, "apiq2-base", "apiq2-cfdisc4L", "test", True)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(A, "agree_conditional.json"))
    args = ap.parse_args()
    out = {}
    for m, sets in {**RUNS, **LLAMA}.items():
        for st, (root, ub, cr, sp, al) in sets.items():
            for name, tag in (("untrained", ub), ("CREDIT", cr)):
                S = pn.rows(root, tag, sp, "syco", al)
                if not S:
                    print(m, st, name, "missing"); continue
                ad = [r for r in S if r["adopt"]]
                s, ci = pn.score(ad), pn.boot(ad)
                out[f"{m}|{st}|{name}"] = dict(ba=round(s[0], 3), ci=ci, n_infl=s[3], n_non=s[4])
                print(f"{m:20s} {st:6s} {name:9s} BA {s[0]:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]  n={s[3]}/{s[4]}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
