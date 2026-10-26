"""Upweight one case type in CREDIT training data by duplication (equivalent to a per-example loss weight).

The paper's upweighting run (Section 6.1, case-type appendix) duplicates the agreeing, non-influential examples with a
correct suggestion (kind == "true", not influenced) W-1 extra times and shuffles with seed 0.
Usage: python -m credit.upweight SRC_NAME OUT_NAME W     (files under $CREDIT_RESULTS/cf_disclose/)"""
import json
import os
import random
import sys

from .common import results_dir


def main():
    src, out, w = sys.argv[1], sys.argv[2], int(sys.argv[3])
    d = os.path.join(results_dir(), "cf_disclose")
    rows = [json.loads(l) for l in open(os.path.join(d, src))]
    hard = [r for r in rows if r["kind"] == "true" and not r["influenced"]]
    out_rows = rows + hard * (w - 1)
    random.Random(0).shuffle(out_rows)
    with open(os.path.join(d, out), "w") as f:
        for r in out_rows:
            f.write(json.dumps(dict(prompt=r["prompt"], completion=r["completion"])) + "\n")
    print(f"[upweight] rows {len(rows)} | upweighted {len(hard)} x{w} -> {len(out_rows)} examples "
          f"(share {len(hard) * w / len(out_rows):.2f})")


if __name__ == "__main__":
    main()
