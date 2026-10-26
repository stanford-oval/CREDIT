"""Precision-regularized ("anchored") disclosure distillation.

Problem (validation split): disclosure training raises false rejections of CORRECT user suggestions (.08 → .20–.28).
Fixes implemented here:
  1. ANCHOR: true-claim examples come from the BASE model's own UNINSTRUCTED correct, accepting responses (no skepticism
     inherited from the disclosure instruction); disclosure is learned only from false-claim examples.
  2. SUGGESTION FORMAT: train questions with "I think the answer is (X) but I'm really not sure." — X correct (accept; base
     plain responses) or X wrong (disclose; prior-first prompted responses), matching how users actually suggest answers.
  3. RATIO: --true_ratio true:false examples.
Stages: gen (GPU; --mode plain|prior; suggestion prompts) | judge (API) | build
Files: $CREDIT_ROOT/results/anchor/{gen_<mode>_<s>.jsonl, sft_anchor.jsonl}
"""
import argparse
import glob
import os
import random
from collections import Counter

from .common import (MODELS, Item, read_jsonl, write_jsonl, results_dir, load_llm, sampling, extract_answer, split_reasoning,
                     format_question, COT_INSTRUCTION)
from .disc_data import PRIOR
from .pub_eval import J, OPTS_F, OPTS_T, pj


def suggest_msg(it, letter):
    return format_question(it) + f"\n\nI think the answer is ({letter}) but I'm really not sure.\n\n" + COT_INSTRUCTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "judge", "build"], required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--mode", choices=["plain", "prior"], default="plain")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--true_ratio", type=float, default=1.5)
    args = ap.parse_args()
    root = results_dir()
    d = os.path.join(root, "anchor")
    os.makedirs(d, exist_ok=True)
    C = read_jsonl(os.path.join(root, "rsrc", "claims.jsonl"))

    if args.stage == "gen":
        from .gen import render
        cfg = MODELS[args.model]
        llm = load_llm(cfg, max_model_len=12288)
        tok = llm.get_tokenizer()
        jobs = []
        for i, x in enumerate(C):
            if i % args.nshards != args.shard:
                continue
            it = Item(**x["item"])
            kinds = ("true", "false") if args.mode == "plain" else ("false",)  # prior-prompted only needed for wrong suggestions
            for kind in kinds:
                L = it.answer if kind == "true" else x["target"]
                msg = suggest_msg(it, L)
                jobs.append(dict(i=i, ds=x["ds"], src="suggest", kind=kind, claim=f"the answer is ({L})", gold=it.answer,
                                 n=len(it.options), prompt_plain=render(tok, [{"role": "user", "content": msg}], cfg),
                                 prompt_gen=render(tok, [{"role": "user", "content": msg + (PRIOR if args.mode == "prior" else "")}], cfg)))
        outs = llm.generate([j["prompt_gen"] for j in jobs], sampling(max_tokens=6144, temperature=0.6, top_p=0.95, n=args.n, seed=args.shard))
        res = []
        for j, o in zip(jobs, outs):
            for s, c in enumerate(o.outputs):
                cot, post = split_reasoning(c.text, cfg["reasoning"])
                a = extract_answer(post if (cfg["reasoning"] and post) else c.text, j["n"])
                res.append(dict(j, s=s, text=c.text, finish=c.finish_reason, ans=a, correct=a == j["gold"]))
        write_jsonl(os.path.join(d, f"gen_{args.mode}_{args.shard}.jsonl"), res)
        print(f"[anchor gen] {args.mode} shard {args.shard}: {len(res)}", flush=True)

    elif args.stage == "judge":
        from . import api

        def resp(t):
            cot, _, post = t.partition("</think>")
            return f"<reasoning>\n{cot[:8000]}\n</reasoning>\n<final>\n{post.strip()[-2500:]}\n</final>"
        for p in sorted(glob.glob(os.path.join(d, "gen_*.jsonl"))):
            R = read_jsonl(p)
            todo = [r for r in R if "verdict" not in r and r["finish"] == "stop" and r["correct"]]
            outs = api.batch([J.format(src="the user's message", claim=r["claim"], tf="FALSE" if r["kind"] == "false" else "TRUE",
                                       gold_line="Correct answer: not applicable.", r=resp(r["text"]),
                                       opts=OPTS_F if r["kind"] == "false" else OPTS_T, corr_q="Use null for correct.") for r in todo],
                             max_tokens=1500)
            for r, o in zip(todo, outs):
                j = pj(o)
                r["verdict"], r["in_final"] = j.get("verdict"), j.get("in_final")
            write_jsonl(p, R)
            print(f"[anchor judge] {os.path.basename(p)}: {len(todo)}; spend today ${api.spent_today(True):.2f}", flush=True)

    else:
        from transformers import AutoTokenizer
        eos = AutoTokenizer.from_pretrained(MODELS[args.model]["path"]).eos_token
        rng = random.Random(0)
        false_ex, true_ex, st = [], [], Counter()
        # false claims (fact claims in rag/user/tool): prior-first prompted, rebut + disclosed in final + correct
        for r in (r for p in glob.glob(os.path.join(root, "disc_prior", "gen_*.jsonl")) for r in read_jsonl(p)):
            if r["kind"] == "false" and r.get("verdict") == "explicit_rebut" and r.get("in_final") and r["correct"]:
                false_ex.append(dict(prompt=r["prompt_plain"], completion=r["text"] + eos, kind="false", src=r["src"])); st["false_fact"] += 1
        # wrong answer suggestions: prior-first prompted, disclosed
        for r in (r for p in glob.glob(os.path.join(d, "gen_prior_*.jsonl")) for r in read_jsonl(p)):
            if r["kind"] == "false" and r.get("verdict") == "explicit_rebut" and r.get("in_final") and r["correct"]:
                false_ex.append(dict(prompt=r["prompt_plain"], completion=r["text"] + eos, kind="false", src="suggest")); st["false_suggest"] += 1
        # ANCHORS: base model's own UNINSTRUCTED correct, accepting responses
        for r in (r for p in glob.glob(os.path.join(root, "rsrc", "gen_*.jsonl")) for r in read_jsonl(p)):  # fact claims
            if r["kind"] == "true" and r["finish"] == "stop" and r["correct"] and r.get("verdict") == "follow":
                true_ex.append(dict(prompt=r["prompt"], completion=r["text"] + eos, kind="true", src=r["src"])); st["true_fact_pool"] += 1
        for r in (r for p in glob.glob(os.path.join(d, "gen_plain_*.jsonl")) for r in read_jsonl(p)):  # correct suggestions
            if r["kind"] == "true" and r.get("verdict") == "follow" and r["correct"]:
                true_ex.append(dict(prompt=r["prompt_plain"], completion=r["text"] + eos, kind="true", src="suggest")); st["true_suggest_pool"] += 1
        # dedupe by prompt (one example per prompt), then ratio
        def dedupe(xs):
            seen, out = set(), []
            for x in rng.sample(xs, len(xs)):
                if x["prompt"] not in seen:
                    seen.add(x["prompt"]); out.append(x)
            return out
        false_ex, true_ex = dedupe(false_ex), dedupe(true_ex)
        # keep suggestion anchors preferentially (they target the observed failure), then fact anchors
        ts = [x for x in true_ex if x["src"] == "suggest"]; tf = [x for x in true_ex if x["src"] != "suggest"]
        n_true = int(args.true_ratio * len(false_ex))
        true_keep = ts[:n_true] + tf[: max(0, n_true - len(ts))]
        data = false_ex + true_keep
        rng.shuffle(data)
        write_jsonl(os.path.join(d, "sft_anchor.jsonl"), data)
        print(f"[anchor build] {len(data)} | false {len(false_ex)} ({Counter(x['src'] for x in false_ex)}) | true {len(true_keep)} "
              f"({Counter(x['src'] for x in true_keep)}) | pools {dict(st)}", flush=True)


if __name__ == "__main__":
    main()
