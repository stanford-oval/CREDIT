"""Multi-source counterfactual relabeling: CREDIT training data with TWO conflicting pieces of supplementary
information per prompt, so self-attribution must name the consequential source, not merely the one that agrees.

Training questions: rl_check/trivia.jsonl (train pool; disjoint from the pub_eval test questions). Each question gets one
ordered pair (W carries the wrong answer X, C carries the correct answer Y) from the TRAINED source types (user
suggestion, tool output, another assistant's answer); question metadata stays held out for evaluation (build_multi).
For each response (2 samples) and each source, leave-one-out consequentiality with 4 re-asks per removal:
  W consequential  = answer gives X  and  re-asks with only C give X at most 1 in 4;  not = answer does not give X, or >= 3 in 4
  C consequential  = answer gives Y  and  re-asks with only W give Y at most 1 in 4;  not analogously
  any target ambiguous -> example dropped
Target = the model's own response with only the RELIED ON line rewritten: it names the consequential source (with its
answer), or "none". Adoption is judged by the LLM answer matcher (llm_match.PROMPT), as for the main training data.
Stages: gen (GPU, sharded) | label (API) | build (mixes with the single-source CREDIT data)
Files: $CREDIT_ROOT/results/cf_multi/{gen_<shard>.jsonl, llm_adopt.json, sft_multi.jsonl}
"""
import argparse
import glob
import json
import os
import random
import re
from collections import Counter

from .common import MODELS, read_jsonl, write_jsonl, results_dir
from .reliance import RELY, RL
from .rl_check import OPEN_INSTR, norm
from .build_multi import compose

TRAIN_SRCS = ("sugg", "tool", "peer")
LINE = {"sugg": "RELIED ON: your suggested answer ({x}) — I would not have given this answer without it.",
        "tool": "RELIED ON: the search tool's result ({x}) — I would not have given this answer without it.",
        "peer": "RELIED ON: the other assistant's answer ({x}) — I would not have given this answer without it."}
NONE_LINE = "RELIED ON: none"


def rows(root):
    rng = random.Random(0)
    pairs = [(a, b) for a in TRAIN_SRCS for b in TRAIN_SRCS if a != b]
    out = []
    for i, x in enumerate(read_jsonl(os.path.join(root, "rl_check", "trivia.jsonl"))):
        X, Y = x["wrong"], x["answer"]
        if not X or not Y or X.strip().lower() == Y.strip().lower():
            continue
        w, c = rng.choice(pairs)
        instr = "\n\n" + OPEN_INSTR
        out.append(dict(qid=f"t{i}", question=x["q"], gold=x["aliases"] + [Y], X=X, Y=Y, w=w, c=c,
                        user=compose(x["q"], instr, {w: X, c: Y}),
                        only_c=compose(x["q"], instr, {c: Y}), only_w=compose(x["q"], instr, {w: X})))
    return out


def final(text):
    _, sep, fin = text.partition("</think>")
    if not sep:
        return None
    m = re.findall(r"final answer\s*:\s*(.+)", fin, re.I)
    return norm(m[-1]) if m else None


def k4(own, re_):
    """True / False / None (ambiguous or unusable), as in the main recipe."""
    v = [r for r in re_ if r is not None]
    if own is None or len(v) < 3:
        return None
    rate = sum(v) / len(v)
    if own and rate <= 0.25:
        return True
    if (not own) or rate >= 0.75:
        return False
    return None


def relabel(text, line):
    cot, sep, fin = text.partition("</think>")
    lines = fin.rstrip().split("\n")
    idx = [k for k, l in enumerate(lines) if RL.search(l)]
    if idx:
        lines[idx[-1]] = line
    else:
        lines.append(line)
    return cot + sep + "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "label", "build"], required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=6144)
    ap.add_argument("--base", default="sft_cf4_llm.jsonl", help="build: single-source CREDIT data to mix with")
    ap.add_argument("--out_name", default="sft_cf4multi.jsonl")
    a = ap.parse_args()
    root = results_dir(); d = os.path.join(root, "cf_multi"); os.makedirs(d, exist_ok=True)
    R = rows(root)
    if a.limit:
        R = R[:a.limit]

    if a.stage == "gen":
        from .common import load_llm, sampling
        from .gen import render
        cfg = MODELS[a.model]
        llm = load_llm(cfg, max_model_len=a.max_new + 2048); tok = llm.get_tokenizer()
        mine = [r for k, r in enumerate(R) if k % a.nshards == a.shard]
        main_o = llm.generate([render(tok, [{"role": "user", "content": r["user"] + RELY}], cfg) for r in mine],
                              sampling(max_tokens=a.max_new, temperature=0.6, top_p=0.95, n=2, seed=40 + a.shard))
        P = [render(tok, [{"role": "user", "content": r[k]}], cfg) for r in mine for k in ("only_c", "only_w")]
        re_o = llm.generate(P, sampling(max_tokens=a.max_new, temperature=0.6, top_p=0.95, n=4, seed=50 + a.shard))
        out = []
        for j, (r, o) in enumerate(zip(mine, main_o)):
            oc, ow = re_o[2 * j], re_o[2 * j + 1]
            out.append(dict(r, texts=[c.text for c in o.outputs], finish=[c.finish_reason for c in o.outputs],
                            only_c_ans=[final(c.text) if c.finish_reason != "length" else None for c in oc.outputs],
                            only_w_ans=[final(c.text) if c.finish_reason != "length" else None for c in ow.outputs]))
        write_jsonl(os.path.join(d, f"gen_{a.shard}.jsonl"), out)
        print(f"[cf_multi gen] shard {a.shard}: {len(out)} prompts", flush=True)
        return

    G = [g for p in sorted(glob.glob(os.path.join(d, "gen_*.jsonl"))) for g in read_jsonl(p)]
    lp = os.path.join(d, "llm_adopt.json")

    def jkey(g, x, ans):
        return f"{g['qid']}|{x}|{ans}"

    if a.stage == "label":
        from . import api
        from .pub_eval import pj
        from .llm_match import PROMPT
        todo = {}
        for g in G:
            gold = "; ".join(g["gold"][:6])
            answers = [final(t) for t, f in zip(g["texts"], g["finish"]) if f != "length"] + g["only_c_ans"] + g["only_w_ans"]
            for ans in {s for s in answers if s}:
                for x in (g["X"], g["Y"]):
                    todo[jkey(g, x, ans)] = PROMPT.format(q=g["question"], x=x, gold=gold, ans=ans[:600])
        res = json.load(open(lp)) if os.path.exists(lp) else {}
        left = [k for k in sorted(todo) if res.get(k) is None]
        print(f"[cf_multi label] {len(todo)} judgments, {len(left)} to do", flush=True)
        for i in range(0, len(left), 4000):
            ch = left[i:i + 4000]
            for k, o in zip(ch, api.batch([todo[k] for k in ch], max_tokens=1500)):
                v = pj(o).get("gives_x")
                if isinstance(v, bool):
                    res[k] = v
            json.dump(res, open(lp, "w"))
            print(f"[cf_multi label] {min(i + 4000, len(left))}/{len(left)} | spend today ${api.spent_today(True):.2f}", flush=True)
        return

    # build
    L = json.load(open(lp))
    adopt = lambda g, x, ans: (L.get(jkey(g, x, ans)) if ans else None)
    st, data = Counter(), []
    for g in G:
        for t, f in zip(g["texts"], g["finish"]):
            if f == "length":
                st["truncated"] += 1; continue
            ans = final(t)
            if ans is None:
                st["no_answer"] += 1; continue
            lw = k4(adopt(g, g["X"], ans), [adopt(g, g["X"], s) for s in g["only_c_ans"]])
            lc = k4(adopt(g, g["Y"], ans), [adopt(g, g["Y"], s) for s in g["only_w_ans"]])
            if lw is None or lc is None:
                st["ambiguous"] += 1; continue
            if lw and lc:
                st["both_consequential?"] += 1; continue  # cannot give X and Y at once; guards matcher errors
            line = LINE[g["w"]].format(x=g["X"]) if lw else LINE[g["c"]].format(x=g["Y"]) if lc else NONE_LINE
            st["w_cons" if lw else "c_cons" if lc else "none"] += 1
            data.append(dict(g, text=t, line=line, cons=bool(lw or lc)))
    print(f"[cf_multi build] {dict(st)}", flush=True)
    from transformers import AutoTokenizer
    from .gen import render
    cfg = MODELS[a.model]; tok = AutoTokenizer.from_pretrained(cfg["path"]); eos = tok.eos_token
    rng = random.Random(0)
    I = [x for x in data if x["cons"]]; N = [x for x in data if not x["cons"]]
    N = rng.sample(N, min(len(N), 2 * len(I)))  # 1:2 consequential : not, as in the main recipe
    multi = [dict(prompt=render(tok, [{"role": "user", "content": x["user"] + RELY}], cfg), completion=relabel(x["text"], x["line"]) + eos,
                  influenced=x["cons"], src=f"multi:{x['w']}+{x['c']}", kind="multi") for x in I + N]
    base = read_jsonl(os.path.join(root, "cf_disclose", a.base))
    out = base + multi
    rng.shuffle(out)
    write_jsonl(os.path.join(d, a.out_name), out)
    print(f"[cf_multi build] multi examples {len(multi)} (consequential {len(I)}, not {len(N)}) + base {len(base)} -> {len(out)} "
          f"| lines: {Counter(x['line'].split('(')[0] for x in I + N).most_common(4)}", flush=True)


if __name__ == "__main__":
    main()
