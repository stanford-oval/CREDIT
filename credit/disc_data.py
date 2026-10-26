"""Disclosure distillation data ("thinking isn't telling"): teach the FINAL ANSWER to state rejected input claims.

Train-split questions with verified false/true claims from rsrc/claims.jsonl (sources rag / user / tool; FaithEval ARC
questions excluded upstream). The model generates WITH a disclosure instruction appended; we keep its own responses that
  false claim: are correct, rebut the claim (rubric v2), and state the rejection in the FINAL answer (in_final)
  true claim : are correct and use/accept the claim (no false rebuttal)
and train on them WITHOUT the instruction (context distillation).
Control arm: the same prompts with the model's own correct responses generated WITHOUT the instruction (rsrc/gen_*.jsonl),
so both arms are self-generated and differ in final-answer disclosure.
Stages: gen (GPU; --shard/--nshards) | judge (API) | build
Files: $CREDIT_ROOT/results/disc/{gen_<s>.jsonl, sft_disc.jsonl, sft_ctrl.jsonl}
"""
import argparse
import glob
import os
import random
from collections import Counter, defaultdict

from .common import (MODELS, Item, read_jsonl, write_jsonl, results_dir, load_llm, sampling, extract_answer, split_reasoning,
                     format_question, COT_INSTRUCTION)
from .ctx_eval import SOURCES
from .pub_eval import J, OPTS_F, OPTS_T, pj

DISC = ("\n\nIf the question, the user, or any provided document or passage contains a claim you conclude is wrong or "
        "inconsistent, say so briefly in your final answer (what was wrong and why). Do not reject claims that are correct.")
PRIOR = ("\n\nFirst, before relying on anything you were given (by the user, or in any document or passage), write down "
         "what YOU believe the answer is from your own knowledge, and how confident you are. Then check each claim you were "
         "given against that belief: accept claims that fit what you know, and treat claims that contradict your confident "
         "knowledge as possibly wrong. Then solve. In your final answer, briefly state any claim you rejected and why. "
         "Do not reject claims that are correct.")
SRCN = {"rag": "the retrieved passage", "user": "the user's message", "tool": "a search-tool result"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "judge", "build"], required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--variant", choices=["disc", "prior"], default="disc", help="which instruction is distilled")
    ap.add_argument("--balance", type=float, default=0.0, help="build: cap true-claim examples at this multiple of false ones (0=no cap)")
    args = ap.parse_args()
    root = results_dir()
    d = os.path.join(root, "disc" if args.variant == "disc" else "disc_prior")
    INSTR = DISC if args.variant == "disc" else PRIOR
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
            for src, tmpl in SOURCES.items():
                for kind in ("false", "true"):
                    msg = tmpl.format(c=x[f"{kind}_claim"]) + format_question(it) + "\n\n" + COT_INSTRUCTION
                    jobs.append(dict(i=i, ds=x["ds"], src=src, kind=kind, claim=x[f"{kind}_claim"], gold=it.answer, n=len(it.options),
                                     prompt_plain=render(tok, [{"role": "user", "content": msg}], cfg),
                                     prompt_disc=render(tok, [{"role": "user", "content": msg + INSTR}], cfg)))
        outs = llm.generate([j["prompt_disc"] for j in jobs], sampling(max_tokens=6144, temperature=0.6, top_p=0.95, n=args.n, seed=args.shard))
        res = []
        for j, o in zip(jobs, outs):
            for s, c in enumerate(o.outputs):
                cot, post = split_reasoning(c.text, cfg["reasoning"])
                a = extract_answer(post if (cfg["reasoning"] and post) else c.text, j["n"])
                res.append(dict(j, s=s, text=c.text, finish=c.finish_reason, ans=a, correct=a == j["gold"]))
        write_jsonl(os.path.join(d, f"gen_{args.shard}.jsonl"), res)
        print(f"[disc gen] shard {args.shard}: {len(res)}", flush=True)

    elif args.stage == "judge":
        from . import api

        def resp(t):
            cot, _, post = t.partition("</think>")
            cot = cot[:7000] + ("\n...\n" + cot[-1000:] if len(cot) > 8000 else cot[7000:])
            return f"<reasoning>\n{cot}\n</reasoning>\n<final>\n{post.strip()[-2500:]}\n</final>"
        for p in sorted(glob.glob(os.path.join(d, "gen_*.jsonl"))):
            R = read_jsonl(p)
            todo = [r for r in R if "verdict" not in r and r["finish"] == "stop" and r["correct"]]
            prompts = [J.format(src=SRCN[r["src"]], claim=r["claim"], tf="FALSE" if r["kind"] == "false" else "TRUE",
                                gold_line="Correct answer: not applicable.", r=resp(r["text"]),
                                opts=OPTS_F if r["kind"] == "false" else OPTS_T, corr_q="Use null for correct.") for r in todo]
            outs = api.batch(prompts, max_tokens=1500)
            for r, o in zip(todo, outs):
                j = pj(o)
                r["verdict"], r["in_final"] = j.get("verdict"), j.get("in_final")
            write_jsonl(p, R)
            print(f"[disc judge] {os.path.basename(p)}: {len(todo)}; spend today ${api.spent_today(True):.2f}", flush=True)

    else:
        from transformers import AutoTokenizer
        eos = AutoTokenizer.from_pretrained(MODELS[args.model]["path"]).eos_token
        R = [r for p in sorted(glob.glob(os.path.join(d, "gen_*.jsonl"))) for r in read_jsonl(p) if r.get("verdict")]
        good = defaultdict(list)
        for r in R:
            ok = (r["kind"] == "false" and r["verdict"] == "explicit_rebut" and r.get("in_final")) or \
                 (r["kind"] == "true" and r["verdict"] == "follow")
            if ok:
                good[(r["i"], r["src"], r["kind"])].append(r)
        plain = defaultdict(list)  # control pool: own correct responses WITHOUT the instruction (realistic-source run)
        for p in sorted(glob.glob(os.path.join(root, "rsrc", "gen_*.jsonl"))):
            for r in read_jsonl(p):
                if r["finish"] == "stop" and r["correct"]:
                    plain[(r["i"], r["src"], r["kind"])].append(r)
        rng = random.Random(0)
        disc, ctrl, st = [], [], Counter()
        for k, rs in good.items():
            if not plain.get(k):
                st["no_control"] += 1
                continue
            g, c = rng.choice(rs), rng.choice(plain[k])
            disc.append(dict(prompt=g["prompt_plain"], completion=g["text"] + eos, kind=k[2], src=k[1]))
            ctrl.append(dict(prompt=c["prompt"], completion=c["text"] + eos, kind=k[2], src=k[1]))
            st[f"{k[2]}_{k[1]}"] += 1
        if args.balance:
            nf = sum(x["kind"] == "false" for x in disc)
            keep = set(rng.sample([i for i, x in enumerate(disc) if x["kind"] == "true"],
                                  min(sum(x["kind"] == "true" for x in disc), int(args.balance * nf))))
            idx = [i for i, x in enumerate(disc) if x["kind"] == "false" or i in keep]
            disc, ctrl = [disc[i] for i in idx], [ctrl[i] for i in idx]
            st["balanced_total"] = len(disc)
        write_jsonl(os.path.join(d, "sft_disc.jsonl"), disc)
        write_jsonl(os.path.join(d, "sft_ctrl.jsonl"), ctrl)
        fin = Counter((r["kind"], r["verdict"], bool(r.get("in_final"))) for r in R)
        print(f"[disc build] {len(disc)} pairs | {dict(st)} | prompted verdicts {dict(fin.most_common(8))}", flush=True)


if __name__ == "__main__":
    main()
