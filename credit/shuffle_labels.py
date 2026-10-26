"""Format-only control: the same SFT data with the reliance labels shuffled across examples (review point (b)).
Each example keeps its prompt and its own response; only the final RELIED ON line is rewritten to a label drawn from a
random permutation of all labels (so the label distribution is unchanged but carries no information about influence).
Usage: python -m credit.shuffle_labels --src sft_cf_llm.jsonl --out sft_cf_shufL.jsonl   (files in $CREDIT_RESULTS/cf_disclose)
"""
import argparse
import os
import random
import re

from .common import read_jsonl, write_jsonl, results_dir
from .cf_disclose import DISCLOSE, NONE_LINE

SUGG = re.compile(r"I think the answer is \(?(.+?)\)? but I'm really not sure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = os.path.join(results_dir(), "cf_disclose")
    R = read_jsonl(os.path.join(d, args.src))
    labels = [r["influenced"] for r in R]
    random.Random(args.seed).shuffle(labels)
    out, skipped = [], 0
    for r, lab in zip(R, labels):
        m = SUGG.search(r["prompt"])
        i = r["completion"].rfind("RELIED ON:")
        if not m or i < 0:
            skipped += 1
            continue
        j = r["completion"].find("\n", i)
        tail = r["completion"][j:] if j >= 0 else r["completion"][i:].split("RELIED ON:", 1)[1]
        if j < 0:  # line runs to the end of sequence token: keep only the eos marker
            tail = r["completion"][r["completion"].rfind("<|"):] if "<|" in r["completion"][i:] else ""
        line = DISCLOSE.format(x=m.group(1)) if lab else NONE_LINE
        out.append(dict(r, completion=r["completion"][:i] + line + tail, influenced=lab, shuffled_from=r["influenced"]))
    write_jsonl(os.path.join(d, args.out), out)
    agree = sum(o["influenced"] == o["shuffled_from"] for o in out) / max(1, len(out))
    print(f"[shuffle] {len(out)} examples ({skipped} skipped) | label kept by chance {agree:.2f} | "
          f"influenced share {sum(o['influenced'] for o in out) / max(1, len(out)):.2f}", flush=True)
    ex = next(o for o in out if o["influenced"])
    print("[shuffle] example tail:", repr(ex["completion"][-160:]), flush=True)


if __name__ == "__main__":
    main()
