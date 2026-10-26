"""Untrained disclosure measurement for API (frontier) models: same items, instruction, re-ask influence labels and
judge as reliance.py, so scores are directly comparable with the open models.

Writes $CREDIT_ROOT/results/<tag>/reliance_<split>.jsonl in reliance.py's format; then run
  reliance --stage judge / judge_src, and reliance --stage select --k4 --bench <b>.
Calls are NOT cached (api.complete caches by prompt, which would collapse the k re-ask samples into one); spend is
tracked through api._record and the daily cap is enforced. Provider-default sampling for every call.
API outputs have no visible reasoning: the final text is stored after a '</think>' marker so that answer extraction and
the RELIED-line parser read the whole output as the final answer.
Usage: python -m credit.api_eval --model gpt-5-mini --tag api-gpt-5-mini --items items_xferc --benches syco,x_sys --split test
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

from . import api
from .common import read_jsonl, write_jsonl, results_dir
from .reliance import RELY, RL
from .system_eval import reask_prompt, is_correct, answer_key


def call(model, user, system, max_tokens, retries=6):
    """One uncached chat completion → (text, finish_reason). ('', 'error') on failure."""
    import requests
    if api.spent_today() >= api.DAILY_USD:
        raise api.BudgetExceeded(f"API daily cap reached: ${api.spent_today():.2f}")
    base, key = api._creds()
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens}
    for a in range(retries):
        try:
            r = requests.post(f"{base}/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=600)
            if r.status_code == 200:
                js = r.json()
                ch = js["choices"][0]
                with api._lock:
                    api._record(model, js.get("usage") or {})
                return ch["message"].get("content") or "", ch.get("finish_reason") or ""
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** a * 2)); continue
            print(f"[api_eval] {model} HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return "", "error"
        except Exception:  # noqa
            time.sleep(min(60, 2 ** a * 2))
    return "", "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--items", default="items")
    ap.add_argument("--benches", default="syco")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items per split")
    args = ap.parse_args()
    root = results_dir()
    benches = set(args.benches.split(","))
    for sp in args.split.split(","):
        items = read_jsonl(os.path.join(root, "pub_eval", f"{args.items}_{sp}.jsonl"))
        by_key = {(x["bench"], x["id"], x["kind"]): x for x in items}
        todo = [x for x in items if x["bench"] in benches and x["kind"] in ("true", "false")]
        if args.limit:
            todo = todo[: args.limit]
        plain = sorted({reask_prompt(x, by_key) for x in todo if reask_prompt(x, by_key)})
        jobs = [(x["user"] + RELY, x.get("system")) for x in todo] + [(p, None) for p in plain for _ in range(args.k)]
        with ThreadPoolExecutor(args.workers) as ex:
            outs = list(ex.map(lambda j: call(args.model, j[0], j[1], args.max_tokens), jobs))
        main_o, re_o = outs[: len(todo)], outs[len(todo):]
        samples = {p: re_o[i * args.k:(i + 1) * args.k] for i, p in enumerate(plain)}
        res, nbad = [], 0
        for x, (t, fr) in zip(todo, main_o):
            text = "</think>\n" + t
            trunc = fr in ("length", "error") or not t.strip()
            nbad += trunc
            m = RL.findall(t)
            rp = reask_prompt(x, by_key)
            ks = [answer_key("</think>\n" + s, x) if (f not in ("length", "error") and s.strip()) else None for s, f in samples.get(rp, [])]
            first = next(((s, f) for s, f in samples.get(rp, []) if f not in ("length", "error") and s.strip()), None)
            res.append(dict(bench=x["bench"], id=x["id"], kind=x["kind"], mc=x["mc"], n=x["n"], gold=x["gold"], claim=x["claim"],
                            user=x["user"], text=text, rely_line=(m[-1].strip()[:1000] if m else None), truncated=trunc,
                            own=is_correct(text, x), own_key=answer_key(text, x),
                            reask=(is_correct("</think>\n" + first[0], x) if first else None),
                            reask_key=(answer_key("</think>\n" + first[0], x) if first else None), reask_keys_k=ks,
                            model=args.model))
        os.makedirs(os.path.join(root, args.tag), exist_ok=True)
        write_jsonl(os.path.join(root, args.tag, f"reliance_{sp}.jsonl"), res)
        print(f"[api_eval] {args.model} {sp}: {len(res)} items, {len(plain)} plain x {args.k} | failed/truncated {nbad} | "
              f"RELIED line {sum(r['rely_line'] is not None for r in res)} | spend today ${api.spent_today(True):.2f}", flush=True)


if __name__ == "__main__":
    main()
