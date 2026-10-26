"""Stage 1: generate CoT + answer from a target model.

Modes:
  plain  — unhinted CoT on each dataset (used by the Lanham battery, transfer, thought anchors)
  hints  — unhinted + 6 hinted variants per item (Chen et al. 2025 hint verbalization)

Output: $CREDIT_ROOT/results/<model>/gen_<mode>_<dataset>.jsonl
"""
import argparse
import os

from .common import (MODELS, load_dataset_items, load_llm, sampling, extract_answer, split_reasoning, write_jsonl, results_dir)
from .hints import HINT_TYPES, build_messages, pick_hint_letter


def render(tok, msgs, cfg):
    kw = {}
    if cfg.get("qwen3_nothink"):
        kw["enable_thinking"] = False
    elif "Qwen3" in cfg["path"]:
        kw["enable_thinking"] = True
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    if cfg["reasoning"] and not s.rstrip().endswith("<think>"):
        s += "<think>\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="plain", choices=["plain", "hints"])
    ap.add_argument("--datasets", default="arc,logiqa,aqua,mmlu")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir")
    ap.add_argument("--tag", default=None, help="results subdir name (defaults to --model)")
    ap.add_argument("--split", default=None, help="dataset split override (e.g. train); appended to output name")
    ap.add_argument("--hint_types", default=None, help="comma list; default all six")
    args = ap.parse_args()

    cfg = MODELS[args.model]
    max_tokens = args.max_tokens or (7168 if cfg["reasoning"] else 2048)
    llm_kw = {}
    if args.adapter:
        llm_kw = dict(enable_lora=True, max_lora_rank=64)
    llm = load_llm(cfg, max_model_len=max_tokens + 2048, **llm_kw)
    tok = llm.get_tokenizer()
    lora = None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("adapter", 1, args.adapter)
    temp = 0.6 if cfg["reasoning"] else 0.7
    sp = sampling(max_tokens=max_tokens, temperature=temp, top_p=0.95, n=args.samples, seed=0)
    outdir = os.path.join(results_dir(), args.tag or args.model)

    for ds in args.datasets.split(","):
        path = os.path.join(outdir, f"gen_{args.mode}_{ds}" + (f"_{args.split}" if args.split else "") + ".jsonl")
        if os.path.exists(path) and not os.environ.get("OVERWRITE"):
            print(f"skip existing {path}", flush=True)
            continue
        items = load_dataset_items(ds, n=args.n, seed=0, split=args.split)
        jobs = []  # (item, condition, hint_letter, messages)
        for it in items:
            jobs.append((it, "none", None, build_messages(it, None, None)))
            if args.mode == "hints":
                h = pick_hint_letter(it)
                for ht in (args.hint_types.split(",") if args.hint_types else HINT_TYPES):
                    jobs.append((it, ht, h, build_messages(it, ht, h)))
        prompts = [render(tok, m, cfg) for _, _, _, m in jobs]
        outs = llm.generate(prompts, sp, lora_request=lora)
        rows = []
        for (it, cond, h, _), p, o in zip(jobs, prompts, outs):
            for k, c in enumerate(o.outputs):
                text = c.text
                cot, post = split_reasoning(text, cfg["reasoning"])
                ans = extract_answer(post, len(it.options)) if cfg["reasoning"] else extract_answer(text, len(it.options))
                if ans is None:
                    ans = extract_answer(text, len(it.options))
                rows.append(dict(
                    item=it.to_dict(), condition=cond, hint=h, sample=k, prompt=p, completion=text,
                    cot=cot, post=post, pred=ans, correct=ans == it.answer,
                    finish=c.finish_reason, n_tokens=len(c.token_ids),
                ))
        write_jsonl(path, rows)
        acc = sum(r["correct"] for r in rows if r["condition"] == "none") / max(1, sum(r["condition"] == "none" for r in rows))
        trunc = sum(r["finish"] == "length" for r in rows) / len(rows)
        print(f"[{args.model}] {ds}: {len(rows)} rows, acc(none)={acc:.3f}, truncated={trunc:.3f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
