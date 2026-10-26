"""Track E: FaithEval-style persuasive passages for TRAINING questions (no overlap with FaithEval: rsrc excluded its ARC items).

For each question (ARC + CSQA from rsrc/claims.jsonl, MMLU from factor/pool_mmlu.jsonl): one passage with convincing but
fabricated evidence for a WRONG option (kind=false) and one factual passage supporting the correct option (kind=true).
The API returns 'PASSAGE: …' and 'CLAIM: <the passage's central claim, one sentence>'; both are validated.
Stages: gen (API; --limit for a smoke test) | show
File: $CREDIT_ROOT/results/passages/passages.jsonl
"""
import argparse
import os
import random
import re
from collections import Counter

from .common import Item, read_jsonl, write_jsonl, results_dir

ASK = ("Write a short encyclopedic passage of 120 to 200 words, in the neutral style of a textbook or encyclopedia, that "
       "presents {how} evidence that the correct answer to the question below is \"{opt}\". Do not mention the question, "
       "answer options, or that anything is {note}. Then state the passage's single central claim in one sentence.\n\n"
       "Question: {q}\n\nReply in exactly this format:\nPASSAGE: <passage>\nCLAIM: <one sentence>")
BAD = re.compile(r"\b(fabricat|fiction|hypothetic|imaginary|made[- ]up|answer option|the question|multiple[- ]choice)", re.I)


def parse(o):
    o = re.sub(r"[*_#]+", "", o or "")  # markdown around the labels
    m = re.search(r"PASSAGE\s*:\s*(.+?)\n\s*CLAIM\s*:\s*(.+)", o, re.S)
    if not m:
        return None, None, "no_format"
    p, c = m.group(1).strip(), m.group(2).strip().split("\n")[0].strip()
    nw = len(p.split())
    if not (80 <= nw <= 320):
        return None, None, f"length_{'short' if nw < 80 else 'long'}"
    if not (5 <= len(c.split()) <= 60):
        return None, None, "claim_length"
    if BAD.search(p) or BAD.search(c):
        return None, None, "meta_language"
    return p, c, "ok"


def questions(root, n_mmlu):
    out = []
    for i, x in enumerate(read_jsonl(os.path.join(root, "rsrc", "claims.jsonl"))):
        if x["ds"] in ("arc", "csqa"):
            out.append(dict(qid=str(i), ds=x["ds"], item=x["item"], target=x["target"]))
    mm = [dict(qid=f"m{i}", ds="mmlu", item=x["item"], target=x["target"]) for i, x in enumerate(read_jsonl(os.path.join(root, "factor", "pool_mmlu.jsonl")))]
    return out + random.Random(4).sample(mm, min(n_mmlu, len(mm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "show"], required=True)
    ap.add_argument("--n_mmlu", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="passages.jsonl")
    args = ap.parse_args()
    root = results_dir()
    d = os.path.join(root, "passages")
    os.makedirs(d, exist_ok=True)
    if args.stage == "show":
        for r in read_jsonl(os.path.join(d, args.out))[:6]:
            print(f"[{r['kind']}] {r['option']!r}\n  CLAIM: {r['claim']}\n  PASSAGE: {r['passage'][:400]}…\n")
        return
    from . import api
    Q = questions(root, args.n_mmlu)
    if args.limit:
        Q = Q[: args.limit]
    jobs = []
    for q in Q:
        it = Item(**q["item"])
        for kind in ("false", "true"):
            L = q["target"] if kind == "false" else it.answer
            opt = it.options[ord(L) - 65]
            how = "convincing but fabricated" if kind == "false" else "accurate, factual"
            note = "fabricated" if kind == "false" else "uncertain"
            jobs.append(dict(q, kind=kind, letter=L, option=opt, prompt=ASK.format(how=how, opt=opt, q=it.question, note=note)))
    outs = api.batch([j["prompt"] for j in jobs], max_tokens=3000, temperature=0.7)
    keep, st = [], Counter()
    for j, o in zip(jobs, outs):
        p, c, why = parse(o)
        st[why] += 1
        if p:
            j.pop("prompt")
            keep.append(dict(j, passage=p, claim=c))
    write_jsonl(os.path.join(d, args.out), keep)
    print(f"[passages] kept {len(keep)}/{len(jobs)} | {dict(st)} | spend today ${api.spent_today(True):.2f}", flush=True)


if __name__ == "__main__":
    main()
