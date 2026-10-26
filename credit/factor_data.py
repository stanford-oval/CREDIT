"""Factorized disclosure training: reject a claim BECAUSE own knowledge contradicts it, and say so.

Disclosure distillation shifted overall trust (knows+false .77→.85 but doesn't-know+true .77→.64). Here the training
data is factorized so the reject/accept decision can only be predicted from (own knowledge × claim truth):
  * KNOWLEDGE: k closed-book samples per train question (no claim); knows = share correct >= --know_thr.
  * PAIRS: the same question with a true and a false claim, both generated under the SAME prior-first prompt (style
    cannot leak the label), trained without the instruction.
  * CELLS: knows+false → override + tell (rebut, in_final, correct); any+true → accept (follow, correct);
    doesn't-know+false → no target. Known questions enter only as complete pairs; doesn't-know+true capped by --unk_ratio.
Sources: fact claims (rag/user/tool) from disc_prior/gen_*; suggestions from anchor/gen_prior_* (false) + this gen (true).
Stages: closed (GPU) | gen (GPU; prior-first, suggestion with a TRUE guess) | judge (API) | build | cells (eval: 2x2 table)
Files: $CREDIT_ROOT/results/factor/{closed_<s>.jsonl, gen_true_<s>.jsonl, sft_factor.jsonl}
"""
import argparse
import glob
import os
import random
from collections import Counter, defaultdict

from .common import (MODELS, Item, read_jsonl, write_jsonl, results_dir, load_llm, sampling, extract_answer, split_reasoning,
                     format_question, COT_INSTRUCTION)
from .disc_data import PRIOR
from .anchor_data import suggest_msg
from .pub_eval import J, OPTS_F, OPTS_T, pj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["closed", "gen", "judge", "build", "cells"], required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--know_thr", type=float, default=1.0, help="knows = closed-book share correct >= this (4/4)")
    ap.add_argument("--unk_ratio", type=float, default=1.0, help="doesn't-know+true examples per known pair")
    ap.add_argument("--tags", default="qwen3-8b,qwen3-8b-pub-distill-prior,qwen3-8b-pub-distill-bal")
    ap.add_argument("--split", default="val,val2")
    ap.add_argument("--pool", choices=["rsrc", "mmlu"], default="rsrc", help="mmlu: harder questions (more UNSURE / confidently-wrong cases), suggestion pairs")
    ap.add_argument("--pool_n", type=int, default=3000)
    ap.add_argument("--out_name", default="sft_factor.jsonl")
    args = ap.parse_args()
    root = results_dir()
    d = os.path.join(root, "factor")
    os.makedirs(d, exist_ok=True)

    if args.stage == "cells":  # per-cell eval on sycophancy-eval: knows = own no-suggestion answer correct
        for tag in args.tags.split(","):
            R = defaultdict(dict)
            for sp in args.split.split(","):
                f = os.path.join(root, tag, "pub_eval.jsonl" if sp == "dev" else f"pub_eval_{sp}.jsonl")
                if not os.path.exists(f):
                    continue
                for r in read_jsonl(f):
                    if r["bench"] == "syco" and r.get("correct") is not None:
                        R[(sp, r["id"])][r["kind"]] = r
            c = Counter()
            for x in R.values():
                if not all(k in x for k in ("plain", "false", "true")):
                    continue
                kn = bool(x["plain"]["correct"])
                for kind in ("false", "true"):
                    cell = ("K" if kn else "U") + kind[0].upper()
                    c[cell] += 1; c[cell + "_ok"] += bool(x[kind]["correct"])
                    c[cell + "_rej"] += x[kind].get("verdict") in ("explicit_rebut", "false_rebut")
            cells = " | ".join(f"{k} acc {c[k+'_ok']/c[k]:.2f} rej {c[k+'_rej']/c[k]:.2f} (n{c[k]})" for k in ("KF", "UT", "KT", "UF") if c[k])
            print(f"{tag:32s} {cells}", flush=True)
        return

    C = read_jsonl(os.path.join(root, "rsrc", "claims.jsonl"))
    pre = ""
    if args.pool == "mmlu":  # MMLU test questions (no overlap with sycophancy-eval / RAMDocs / FaithEval-ARC / PCBench)
        pre = "m"
        pp = os.path.join(d, "pool_mmlu.jsonl")
        if not os.path.exists(pp):
            from .common import load_dataset_items, LETTERS
            its = load_dataset_items("mmlu")
            prng = random.Random(5)
            its = prng.sample(its, args.pool_n)
            write_jsonl(pp, [dict(ds="mmlu", item=it.__dict__, target=prng.choice([L for L in LETTERS[:len(it.options)] if L != it.answer]))
                             for it in its])
        C = read_jsonl(pp)
    if args.stage in ("closed", "gen"):
        from .gen import render
        cfg = MODELS[args.model]
        llm = load_llm(cfg, max_model_len=12288)
        tok = llm.get_tokenizer()
        jobs = []
        for i, x in enumerate(C):
            if i % args.nshards != args.shard:
                continue
            it = Item(**x["item"])
            if args.stage == "closed":
                msg = format_question(it) + "\n\n" + COT_INSTRUCTION
                jobs.append(dict(i=pre + str(i) if pre else i, gold=it.answer, n=len(it.options), prompt_gen=render(tok, [{"role": "user", "content": msg}], cfg)))
            elif pre:  # mmlu pool: both a correct and a wrong suggestion, same prior-first prompt
                for kind in ("true", "false"):
                    L = it.answer if kind == "true" else x["target"]
                    msg = suggest_msg(it, L)
                    jobs.append(dict(i=pre + str(i), ds="mmlu", src="suggest", kind=kind, claim=f"the answer is ({L})", gold=it.answer,
                                     n=len(it.options), prompt_plain=render(tok, [{"role": "user", "content": msg}], cfg),
                                     prompt_gen=render(tok, [{"role": "user", "content": msg + PRIOR}], cfg)))
            else:
                msg = suggest_msg(it, it.answer)
                jobs.append(dict(i=i, ds=x["ds"], src="suggest", kind="true", claim=f"the answer is ({it.answer})", gold=it.answer,
                                 n=len(it.options), prompt_plain=render(tok, [{"role": "user", "content": msg}], cfg),
                                 prompt_gen=render(tok, [{"role": "user", "content": msg + PRIOR}], cfg)))
        k = args.k if args.stage == "closed" else args.n
        outs = llm.generate([j["prompt_gen"] for j in jobs], sampling(max_tokens=6144, temperature=0.6, top_p=0.95, n=k, seed=10 + args.shard))
        res = []
        for j, o in zip(jobs, outs):
            for s, c in enumerate(o.outputs):
                cot, post = split_reasoning(c.text, cfg["reasoning"])
                a = extract_answer(post if (cfg["reasoning"] and post) else c.text, j["n"])
                rec = dict(j, s=s, ans=a, correct=a == j["gold"], finish=c.finish_reason)
                if args.stage == "gen":
                    rec["text"] = c.text
                rec.pop("prompt_gen")
                res.append(rec)
        name = (f"closed_{pre}{args.shard}.jsonl" if args.stage == "closed" else
                (f"gen_mmlu_{args.shard}.jsonl" if pre else f"gen_true_{args.shard}.jsonl"))
        write_jsonl(os.path.join(d, name), res)
        print(f"[factor {args.stage}] shard {args.shard}: {len(res)}", flush=True)

    elif args.stage == "judge":
        from . import api

        def resp(t):
            cot, _, post = t.partition("</think>")
            return f"<reasoning>\n{cot[:8000]}\n</reasoning>\n<final>\n{post.strip()[-2500:]}\n</final>"
        for p in sorted(glob.glob(os.path.join(d, "gen_true_*.jsonl")) + glob.glob(os.path.join(d, "gen_mmlu_*.jsonl"))):
            R = read_jsonl(p)
            todo = [r for r in R if "verdict" not in r and r["finish"] == "stop" and r["correct"]]
            outs = api.batch([J.format(src="the user's message", claim=r["claim"], tf="TRUE" if r["kind"] == "true" else "FALSE",
                                       gold_line="Correct answer: not applicable.", r=resp(r["text"]),
                                       opts=OPTS_T if r["kind"] == "true" else OPTS_F, corr_q="Use null for correct.") for r in todo], max_tokens=1500)
            for r, o in zip(todo, outs):
                j = pj(o)
                r["verdict"], r["in_final"] = j.get("verdict"), j.get("in_final")
            write_jsonl(p, R)
            print(f"[factor judge] {os.path.basename(p)}: {len(todo)}; spend today ${api.spent_today(True):.2f}", flush=True)

    else:  # build
        from transformers import AutoTokenizer
        eos = AutoTokenizer.from_pretrained(MODELS[args.model]["path"]).eos_token
        rng = random.Random(0)
        cb = defaultdict(list)
        for p in glob.glob(os.path.join(d, "closed_*.jsonl")):
            for r in read_jsonl(p):
                cb[r["i"]].append(bool(r["correct"]))
        know = {i: sum(v) / len(v) for i, v in cb.items()}
        cand = defaultdict(list)  # (i, src, kind) -> good responses, all generated under the prior-first prompt
        srcs = [os.path.join(root, "disc_prior", "gen_*.jsonl"), os.path.join(d, "gen_true_*.jsonl"), os.path.join(root, "anchor", "gen_prior_*.jsonl"),
                os.path.join(d, "gen_mmlu_*.jsonl")]
        for r in (r for g in srcs for p in glob.glob(g) for r in read_jsonl(p)):
            if r.get("finish") != "stop" or not r.get("correct") or not r.get("verdict"):
                continue
            ok = (r["kind"] == "false" and r["verdict"] == "explicit_rebut" and r.get("in_final")) or \
                 (r["kind"] == "true" and r["verdict"] == "follow")
            if ok:
                cand[(r["i"], r["src"], r["kind"])].append(r)
        pairs, unk, st = [], [], Counter()
        for (i, src, kind) in list(cand):
            if kind != "true" or i not in know:
                continue
            if know[i] >= args.know_thr:
                if cand.get((i, src, "false")):
                    pairs.append((rng.choice(cand[(i, src, "true")]), rng.choice(cand[(i, src, "false")]))); st["KT+KF_pair"] += 1
                else:
                    st["K_no_false_target"] += 1
            elif know[i] > 0:  # genuinely UNSURE (inconsistent closed-book answers): use a correct claim
                unk.append(rng.choice(cand[(i, src, "true")])); st["UT_pool"] += 1
            else:  # consistently WRONG: from the inside this looks like "knows" → reserved for the verify target, not trained here
                st["confidently_wrong_skipped"] += 1
        rng.shuffle(unk)
        unk = unk[: int(args.unk_ratio * len(pairs))]
        ex = lambda r, cell: dict(prompt=r["prompt_plain"], completion=r["text"] + eos, kind=r["kind"], src=r["src"], cell=cell)
        data = [ex(t, "KT") for t, _ in pairs] + [ex(f, "KF") for _, f in pairs] + [ex(u, "UT") for u in unk]
        rng.shuffle(data)
        write_jsonl(os.path.join(d, args.out_name), data)
        kv = list(know.values())
        print(f"[factor build] {len(data)} | cells {Counter(x['cell'] for x in data)} | src {Counter(x['src'] for x in data)} | {dict(st)} | "
              f"knows>={args.know_thr}: {sum(v >= args.know_thr for v in kv)}/{len(kv)}", flush=True)


if __name__ == "__main__":
    main()
