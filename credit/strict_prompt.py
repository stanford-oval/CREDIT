"""Stricter-instruction prompting baseline (Appendix: prompting baselines, Table: strict).
Untrained models answer with the stricter instruction (`reliance --variant strict`) on the test, fresh, and large
sets. Re-asks are reused from each model's untrained run (identical prompts without any instruction), so the influence
labels are shared and every comparison is paired. With --label, this copies the re-asks and runs the LLM answer judge
(API); without it, the table is recomputed from saved labels (CPU).
Usage: python -m credit.strict_prompt [--label] [--out strict_prompt.json]
"""
import argparse
import json
import os
import subprocess

import credit.paper_numbers as pn
from credit.common import read_jsonl, write_jsonl

A = os.environ.get("CREDIT_ROOT", ".")
R8, R14, RD = f"{A}/results", f"{A}/results_q14", f"{A}/results_r1q"
RUNS = [  # model, set, root, strict tag, split, untrained tag, CREDIT tag, include TruthfulQA
    ("Qwen3-8B", "test", R8, "rely-strict", "test", "rely-base", "rely-cfdisc4L", False),
    ("Qwen3-8B", "fresh", R8, "rely-strict", "test2", "rely-base", "t2-cfdisc4L", False),
    ("Qwen3-8B", "large", R8, "apiq2-strict", "test", "apiq2-base", "apiq2-cfdisc4L", True),
    ("Qwen3-14B", "test", R14, "rely-strict", "test", "rely-base", "rely-cfdisc4L", False),
    ("Qwen3-14B", "fresh", R14, "rely-strict", "test2", "rely-base", "rely-cfdisc4L", False),
    ("Qwen3-14B", "large", R14, "apiq2-strict", "test", "apiq2-base", "apiq2-cfdisc4L", True),
    ("DeepSeek-R1-7B", "test", RD, "rely-strict", "test", "rely-base", "rely-cfdisc4L", False),
    ("DeepSeek-R1-7B", "fresh", RD, "rely-strict", "test2", "t2-base", "t2-cfdisc4L", False),
    ("DeepSeek-R1-7B", "large", RD, "apiq2-strict", "test", "apiq2-base", "apiq2-cfdisc4L", True),
]


def label(root, tag, sp, base, all_src):
    """Copy the untrained run's re-asks into the strict run, then judge the answers with the LLM answer judge."""
    p = f"{root}/{tag}/reliance_{sp}.jsonl"
    B = {(r["bench"], r["id"], r["kind"]): r for r in read_jsonl(f"{root}/{base}/reliance_{sp}.jsonl")}
    R = read_jsonl(p)
    for r in R:
        b = B.get((r["bench"], r["id"], r["kind"]))
        if b and r["bench"] == "syco":
            for k in ("reask", "reask_key", "reask_keys_k"):
                r[k] = b.get(k)
    write_jsonl(p, R)
    cmd = ["python", "-m", "credit.llm_match", "--stage", "match", "--root", root, "--tags", tag, "--benches", "syco", "--split", sp]
    subprocess.run(cmd + (["--all_sources"] if all_src else []), check=True, env=dict(os.environ, CREDIT_RESULTS=root))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="store_true", help="copy re-asks and run the LLM answer judge first (API)")
    ap.add_argument("--out", default=os.path.join(A, "strict_prompt.json"))
    args = ap.parse_args()
    out = {}
    for m, st, root, tag, sp, base, cr, al in RUNS:
        tag, base = pn.resolve(root, tag, sp), pn.resolve(root, base, sp)  # fresh-set runs may be named t2-*
        if not os.path.exists(f"{root}/{tag}/reliance_{sp}.jsonl"):
            print(m, st, "missing"); continue
        if args.label:
            label(root, tag, sp, base, al)
        S = pn.rows(root, tag, sp, "syco", al); s = pn.score(S); ci = pn.boot(S)
        rec = dict(ba=round(s[0], 3), ci=ci, attr_infl=round(s[1], 3), attr_non=round(s[2], 3), n_infl=s[3], n_non=s[4])
        for name, t in (("untrained", base), ("CREDIT", cr)):
            T = pn.rows(root, t, sp, "syco", al)
            if T:
                rec[f"vs_{name}"] = dict(other=round(pn.score(T)[0], 3), diff=pn.paired(S, T))
        out[f"{m}|{st}"] = rec
        print(f"{m:15s} {st:6s} BA {rec['ba']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] ({rec['attr_infl']:.0%} / {rec['attr_non']:.0%}) | "
              + " | ".join(f"{k[3:]} {v['other']:.2f} ({v['diff'][0]:+.2f})" for k, v in rec.items() if k.startswith("vs_")), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
