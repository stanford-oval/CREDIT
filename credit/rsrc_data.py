"""Realistic-source Rebut-or-Follow training data (on-policy rejection sampling; nothing inserted).

TRAIN-split questions (arc, csqa, logiqa, aqua; FaithEval's ARC ids excluded) get one claim in the INPUT from a retrieved
passage / user message / tool output: a verified FALSE claim (pushes toward a wrong option) or a verified TRUE claim.
The model itself answers (n samples). A judge labels each response (follow / explicit_rebut / ignore; follow / false_rebut /
ignore) and correctness. Two SFT sets over the SAME prompts (they differ only in disclosure):
  rsrc_rof.jsonl     false claim → the model's own CORRECT + EXPLICIT_REBUT response; true claim → CORRECT + FOLLOW
  rsrc_silent.jsonl  false claim → the model's own CORRECT + IGNORE (silent override) response; true claim → identical
Only prompts that have BOTH a correct explicit and a correct silent response are used for the false-claim part.
Stages: claims (API) | gen (GPU; --shard i --nshards k) | judge (API) | build
Files: $CREDIT_ROOT/results/rsrc/{claims.jsonl, gen_<shard>.jsonl, sft_rof.jsonl, sft_silent.jsonl}
"""
import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict

from .common import (MODELS, LETTERS, Item, load_dataset_items, read_jsonl, write_jsonl, results_dir, load_llm, sampling,
                     extract_answer, split_reasoning, format_question, COT_INSTRUCTION)
from .ctx_eval import GEN_CLAIMS, AUDIT, SOURCES, SRC_NAME, J_FALSE, J_TRUE, pj

B = os.path.join(os.environ.get("CREDIT_ROOT", "."), "data/bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["claims", "gen", "judge", "build"], required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--per_ds", type=int, default=300)
    ap.add_argument("--datasets", default="arc,csqa,logiqa,aqua")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    root = results_dir()
    d = os.path.join(root, "rsrc")
    os.makedirs(d, exist_ok=True)
    cp = os.path.join(d, "claims.jsonl")

    if args.stage == "claims":
        from . import api
        # held-out guarantee: FaithEval-counterfactual reuses ARC-Challenge question text; exclude those questions by text
        fe = {json.loads(l)["question"].strip().lower() for l in open(f"{B}/Salesforce__FaithEval-counterfactual-v1.0__test.jsonl")}
        rng = random.Random(1)
        items = []
        for ds in args.datasets.split(","):
            its = [it for it in load_dataset_items(ds, n=None, seed=0, split="train") if it.question.strip().lower() not in fe]
            for it in rng.sample(its, min(len(its), int(args.per_ds * 1.5))):
                tgt = rng.choice([l for l in LETTERS[: len(it.options)] if l != it.answer])
                items.append(dict(ds=ds, item=it.to_dict(), target=tgt))
        outs = api.batch([GEN_CLAIMS.format(q=format_question(Item(**x["item"])), gold=x["item"]["answer"], target=x["target"])
                          for x in items], model="gemini-3.8-flash", max_tokens=3000, temperature=0.3)
        for x, o in zip(items, outs):
            j = pj(o)
            x["false_claim"], x["true_claim"] = (j.get("false_claim") or "").strip(), (j.get("true_claim") or "").strip()
        items = [x for x in items if x["false_claim"] and x["true_claim"]]
        aud = api.batch([AUDIT.format(q=format_question(Item(**x["item"])), s=x[k]) for x in items for k in ("false_claim", "true_claim")],
                        model="gpt-5-mini", max_tokens=3000)
        keep, per = [], Counter()
        for k, x in enumerate(items):
            if pj(aud[2 * k]).get("true") is False and pj(aud[2 * k + 1]).get("true") is True and per[x["ds"]] < args.per_ds:
                keep.append(x)
                per[x["ds"]] += 1
        write_jsonl(cp, keep)
        print(f"[rsrc claims] kept {len(keep)} {dict(per)}; spend today ${api.spent_today(True):.2f}", flush=True)

    elif args.stage == "gen":
        from .gen import render
        cfg = MODELS[args.model]
        llm = load_llm(cfg, max_model_len=12288)
        tok = llm.get_tokenizer()
        C = read_jsonl(cp)
        jobs = []
        for i, x in enumerate(C):
            if i % args.nshards != args.shard:
                continue
            it = Item(**x["item"])
            for src, tmpl in SOURCES.items():
                for kind in ("false", "true"):
                    claim = x[f"{kind}_claim"]
                    msg = tmpl.format(c=claim) + format_question(it) + "\n\n" + COT_INSTRUCTION
                    jobs.append(dict(i=i, ds=x["ds"], src=src, kind=kind, claim=claim, target=x["target"], gold=it.answer,
                                     n=len(it.options), prompt=render(tok, [{"role": "user", "content": msg}], cfg)))
        outs = llm.generate([j["prompt"] for j in jobs], sampling(max_tokens=4096, temperature=0.6, top_p=0.95, n=args.n, seed=args.shard))
        res = []
        for j, o in zip(jobs, outs):
            for s, c in enumerate(o.outputs):
                cot, post = split_reasoning(c.text, cfg["reasoning"])
                a = extract_answer(post if (cfg["reasoning"] and post) else c.text, j["n"])
                res.append(dict(j, s=s, text=c.text, finish=c.finish_reason, ans=a, correct=a == j["gold"]))
        write_jsonl(os.path.join(d, f"gen_{args.shard}.jsonl"), res)
        print(f"[rsrc gen] shard {args.shard}: {len(res)}", flush=True)

    elif args.stage == "judge":
        from . import api
        for p in sorted(glob.glob(os.path.join(d, "gen_*.jsonl"))):
            R = read_jsonl(p)
            todo = [r for r in R if "verdict" not in r and r["finish"] == "stop" and r["correct"]]  # only correct ones are used
            resp = lambda t: t[:7000] + ("\n...\n" + t[-1500:] if len(t) > 8500 else t[7000:])
            outs = api.batch([(J_FALSE if r["kind"] == "false" else J_TRUE).format(src=SRC_NAME[r["src"]], c=r["claim"], r=resp(r["text"]))
                              for r in todo], max_tokens=1500)
            for r, o in zip(todo, outs):
                r["verdict"] = pj(o).get("verdict")
            write_jsonl(p, R)
            print(f"[rsrc judge] {os.path.basename(p)}: {len(todo)}; spend today ${api.spent_today(True):.2f}", flush=True)

    else:
        from transformers import AutoTokenizer
        eos = AutoTokenizer.from_pretrained(MODELS[args.model]["path"]).eos_token
        R = [r for p in sorted(glob.glob(os.path.join(d, "gen_*.jsonl"))) for r in read_jsonl(p) if r.get("verdict")]
        by = defaultdict(list)
        for r in R:
            by[(r["i"], r["src"], r["kind"])].append(r)
        rng = random.Random(0)
        rof, sil, st = [], [], Counter()
        for (i, src, kind), rs in by.items():
            if kind == "false":
                ex = [r for r in rs if r["verdict"] == "explicit_rebut"]
                ig = [r for r in rs if r["verdict"] == "ignore"]
                if ex and ig:  # matched: same prompt in both arms
                    rof.append(dict(prompt=ex[0]["prompt"], completion=ex[0]["text"] + eos, kind="false_explicit", src=src))
                    sil.append(dict(prompt=ig[0]["prompt"], completion=ig[0]["text"] + eos, kind="false_silent", src=src))
                    st["false_matched"] += 1
                st[f"false_has_explicit"] += bool(ex)
                st[f"false_has_ignore"] += bool(ig)
            else:
                fo = [r for r in rs if r["verdict"] == "follow"]
                if fo:
                    e = dict(prompt=fo[0]["prompt"], completion=fo[0]["text"] + eos, kind="true_follow", src=src)
                    rof.append(e)
                    sil.append(e)
                    st["true_follow"] += 1
        # keep the true-claim share comparable to the false-claim share
        tf = [x for x in rof if x["kind"] == "true_follow"]
        nf = st["false_matched"]
        keep_t = set(id(x) for x in rng.sample(tf, min(len(tf), nf)))
        rof = [x for x in rof if x["kind"] != "true_follow" or id(x) in keep_t]
        sil = [x for x in sil if x["kind"] != "true_follow" or id(x) in keep_t]
        write_jsonl(os.path.join(d, "sft_rof.jsonl"), rof)
        write_jsonl(os.path.join(d, "sft_silent.jsonl"), sil)
        print(f"[rsrc build] rof {len(rof)} silent {len(sil)} | {dict(st)} | verdicts "
              f"{Counter((r['kind'], r['verdict']) for r in R)}", flush=True)


if __name__ == "__main__":
    main()
