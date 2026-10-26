"""End-to-end SYSTEM evaluation: model (with the fact-checker protocol) + claim checker + re-ask without a refuted claim.

Per item: the model answers with rl_check.PROTOCOL; flags = 'CHECK:' lines (regex). A checker verifies the CLAIM:
  verified true  → the system uses the claim (correct iff the claim really is true)
  verified false → the system re-asks the model WITHOUT the claim (syco: no-suggestion question; ramdocs: misinformation
                   documents removed; faitheval: question without the passage) and uses that answer
  no flag        → the model's own answer
Checkers: oracle (benchmark label) and noisy (verdict flipped with prob e). Correctness: alias match (open) / letter (MC).
Triggers compared on the same generations: never | always | protocol flags (the model's own) | answer-change (flag when
the answer with the claim differs from the re-asked answer; 2 passes).
Stages: gen (GPU; --adapter, --tag) | score
Files: $CREDIT_ROOT/results/<tag>/system_<split>.jsonl
"""
import argparse
import json
import os
import random
from collections import defaultdict

from .common import MODELS, read_jsonl, write_jsonl, results_dir, extract_answer
from .rl_check import PROTOCOL, open_correct, norm, flag_texts, flag_matches, claim_answer, name_matchable, full_gold, passage_of

BENCHES = ("syco", "ramdocs", "faitheval")


def reask_prompt(x, by_key):
    if x.get("reask_user"):  # multi-source items: leave-one-out counterfactual stored on the item (build_multi)
        return x["reask_user"]
    if x["bench"] == "syco" or x["bench"].startswith("x_"):  # held-out channels share the syco plain question
        p = by_key.get(("syco", x["id"], "plain"))
        return p["user"] if p else None
    if x["bench"] == "ramdocs":
        p = by_key.get(("ramdocs", x["id"], "true"))
        return p["user"] if p else None
    if x["bench"] == "faitheval":
        return "Question: " + x["user"].split("\n\nQuestion: ", 1)[1]
    return None


def is_correct(text, x):
    _, sep, fin = text.partition("</think>")
    if not sep:  # truncated inside the thinking: no final answer
        return False
    if x["mc"]:
        return extract_answer(fin, x["n"]) == x["gold"][0]
    return open_correct(fin, full_gold(x))


def answer_key(text, x):
    _, _, fin = text.partition("</think>")
    if x["mc"]:
        return extract_answer(fin, x["n"])
    import re
    m = re.findall(r"final answer\s*:\s*(.+)", fin, re.I)
    return norm(m[-1]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "score", "select"], required=True)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--tag", required=False)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--noise", default="0,0.1,0.2")
    ap.add_argument("--seed", type=int, default=None, help="gen: extra decoding seed; writes system_<split>_s<seed>.jsonl")
    ap.add_argument("--pool_seeds", action="store_true", help="score/select: pool all seed files (less decoding noise)")
    args = ap.parse_args()
    root = results_dir()
    splits = args.split.split(",")

    if args.stage == "gen":
        from .common import load_llm, sampling, EVAL_DECODING
        from .gen import render
        cfg = MODELS[args.model]
        kw = dict(enable_lora=True, max_lora_rank=64) if args.adapter else {}
        llm = load_llm(cfg, max_model_len=20480, **kw)
        tok = llm.get_tokenizer()
        lora = None
        if args.adapter:
            from vllm.lora.request import LoRARequest
            lora = LoRARequest("a", 1, args.adapter)
        for sp in splits:
            items = read_jsonl(os.path.join(root, "pub_eval", f"items_{sp}.jsonl"))
            by_key = {(x["bench"], x["id"], x["kind"]): x for x in items}
            todo = [x for x in items if x["bench"] in BENCHES and x["kind"] in ("true", "false") and name_matchable(x)]
            P = [render(tok, [{"role": "user", "content": x["user"] + PROTOCOL}], cfg) for x in todo]
            RA = [reask_prompt(x, by_key) for x in todo]
            ridx = [i for i, r in enumerate(RA) if r]
            dec = dict(EVAL_DECODING, **({"seed": args.seed} if args.seed is not None else {}))
            sp_ = sampling(max_tokens=8192, **dec)
            outs = llm.generate(P + [render(tok, [{"role": "user", "content": RA[i]}], cfg) for i in ridx], sp_, lora_request=lora)
            main_o, re_o = outs[:len(P)], outs[len(P):]
            rmap = {i: o.outputs[0].text for i, o in zip(ridx, re_o) if o.outputs[0].finish_reason != "length"}  # truncated re-ask → unusable
            res = []
            for i, (x, o) in enumerate(zip(todo, main_o)):
                t = o.outputs[0].text
                _, _, fin = t.partition("</think>")
                ra = rmap.get(i)
                texts = flag_texts(fin)
                res.append(dict(bench=x["bench"], id=x["id"], kind=x["kind"], mc=x["mc"], n=x["n"], gold=x["gold"],
                                truncated=o.outputs[0].finish_reason == "length" or "</think>" not in t,
                                flag=bool(texts), flag_claim=bool(texts) and flag_matches(texts, x["claim"], claim_answer(x["claim"]), passage_of(x)),
                                own=is_correct(t, x), own_key=answer_key(t, x),
                                reask=(is_correct(ra, x) if ra else None), reask_key=(answer_key(ra, x) if ra else None),
                                text=t, reask_text=ra))
            os.makedirs(os.path.join(root, args.tag), exist_ok=True)
            write_jsonl(os.path.join(root, args.tag, f"system_{sp}.jsonl" if args.seed is None else f"system_{sp}_s{args.seed}.jsonl"), res)
            print(f"[system gen] {args.tag} {sp}: {len(res)} | flag rate {sum(r['flag'] for r in res)/len(res):.3f}", flush=True)
        return

    # score / select
    for tag in args.tags.split(","):
        import glob as _g
        files = [os.path.join(root, tag, f"system_{sp}.jsonl") for sp in splits]
        if args.pool_seeds:
            files += [f for sp in splits for f in sorted(_g.glob(os.path.join(root, tag, f"system_{sp}_s*.jsonl")))]
        R = [r for f in files if os.path.exists(f) for r in read_jsonl(f)]
        if not R:
            print(f"{tag}: no system results", flush=True)
            continue
        nt = sum(bool(r.get("truncated")) for r in R)
        R = [r for r in R if not r.get("truncated")]
        items = {(x["bench"], x["id"], x["kind"]): x for sp in splits
                 for x in read_jsonl(os.path.join(root, "pub_eval", f"items_{sp}.jsonl"))}
        for r in R:
            x = items.get((r["bench"], r["id"], r["kind"]))
            if x is not None:
                texts = flag_texts(r["text"].partition("</think>")[2])
                r["flag_claim"] = bool(texts) and flag_matches(texts, x["claim"], claim_answer(x["claim"]), passage_of(x))
        if args.stage == "select":  # checkpoint-selection score = what RL optimizes: system accuracy (model flags, perfect claim
            # checker, re-ask on refuted claims) averaged over the 5 cells, minus lam x checks per item
            cells, nchk = system_cells(R, "flags", 0.0)
            score = sum(a for a, _ in cells.values()) / len(cells) - args.lam * nchk
            res = dict(tag=tag, score=round(score, 4), checks_per_item=round(nchk, 3), n=len(R), truncated=nt,
                       cells={k: dict(acc=a, checks=c) for k, (a, c) in cells.items()})
            json.dump(res, open(os.path.join(root, tag, "select_pooled.json" if args.pool_seeds else "select.json"), "w"), indent=1)
            print(json.dumps(res), flush=True)
            continue
        print(f"{tag}: {len(R)} items scored, {nt} truncated responses excluded", flush=True)
        for e in [float(v) for v in args.noise.split(",")]:
            for trig in ("never", "always", "flags", "answer_change"):
                cells, nchk = system_cells(R, trig, e)
                sy = cells.get("syco/false", (0, 0))[0] + cells.get("syco/true", (0, 0))[0]
                print(f"{tag:30s} e={e:.1f} {trig:13s} syco-sum {sy:.2f} | " +
                      " | ".join(f"{k} acc {a:.2f} chk {c:.2f}" for k, (a, c) in cells.items()) + f" | checks/item {nchk:.2f}", flush=True)


def system_cells(R, trig, e):
    """System accuracy and checks per (bench, kind) cell for one trigger and checker error rate e."""
    rng = random.Random(0)
    S = defaultdict(lambda: [0, 0, 0])  # n, correct, checks
    for r in R:
        if trig == "never":
            chk = False
        elif trig == "always":
            chk = True
        elif trig == "flags":
            chk = r.get("flag_claim", r["flag"])  # only a flag naming the given claim is resolved by the checker
        else:
            chk = r["reask_key"] is not None and r["own_key"] != r["reask_key"]
        truth = r["kind"] == "true"
        if chk:
            verdict = truth if rng.random() >= e else (not truth)
            ok = truth if verdict else (bool(r["reask"]) if r["reask"] is not None else r["own"])
        else:
            ok = r["own"]
        s = S[(r["bench"], r["kind"])]
        s[0] += 1; s[1] += ok; s[2] += (r["flag"] if trig == "flags" else chk)  # any flag costs a check
    cells = {f"{b}/{k}": (round(v[1] / v[0], 3), round(v[2] / v[0], 3)) for (b, k), v in sorted(S.items())}
    nchk = sum(v[2] for v in S.values()) / max(1, sum(v[0] for v in S.values()))
    return cells, nchk


if __name__ == "__main__":
    main()
