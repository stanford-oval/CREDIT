"""Answer matching on the full text of long frontier answers (appendix "Frontier Model Evaluation Settings").

The answer judge normally sees a truncated answer. This re-judges every frontier answer longer than 600 characters
(final answer and counterfactual answers) with its full text, recomputes the influence labels, and prints each model's
balanced accuracy before and after. Runs on the large set (results_api/api2-*).
Usage: python -m credit.rejudge_full"""
import glob
import json
import os

from . import api
from . import llm_match as lm
from .pub_eval import pj
from .rl_check import full_gold


def score(S):
    I = [d for l, d in S if l]; N = [d for l, d in S if not l]
    return (sum(I) / len(I) + 1 - sum(N) / len(N)) / 2, sum(I) / len(I), sum(N) / len(N), len(I)


def main():
    lm.ALL_SOURCES = True
    RA = os.path.join(os.environ.get("CREDIT_ROOT", "."), "results_api")
    todo, data = {}, {}
    for d in sorted(glob.glob(f"{RA}/api2-*")):
        tag = os.path.basename(d); L = json.load(open(d + "/llm_adopt_test.json"))
        R = lm.items(RA, tag, "test", "syco"); data[tag] = (R, L)
        for r in R:
            g = "; ".join(full_gold(r)[:6]); x = lm.x_of(r)
            for a in [r.get("own_key")] + list(r.get("reask_keys_k") or []):
                if a and len(a) > 600:
                    todo[(r["id"], x, a)] = lm.PROMPT.format(q=r["id"], x=x, gold=g, ans=a)
    keys = sorted(todo); print("long unique answers:", len(keys), flush=True)
    res = {}
    for _ in range(2):
        left = [k for k in keys if res.get(k) is None]
        for k, o in zip(left, api.batch([todo[k] for k in left], max_tokens=1500)):
            v = pj(o).get("gives_x"); res[k] = v if isinstance(v, bool) else None
    print("unparsed:", sum(v is None for v in res.values()))
    flips = changed = 0
    for tag, (R, L) in data.items():
        a, b = [], []
        for r in R:
            v = L.get(lm.key(r))
            if not v:
                continue
            x = lm.x_of(r); own, re_ = v["own"], list(v["reask"]); ok = r.get("own_key")
            if ok and len(ok) > 600 and res.get((r["id"], x, ok)) is not None:
                flips += res[(r["id"], x, ok)] != own; own = res[(r["id"], x, ok)]
            for i, k in enumerate(r.get("reask_keys_k") or []):
                if k and len(k) > 600 and res.get((r["id"], x, k)) is not None:
                    flips += res[(r["id"], x, k)] != re_[i]; re_[i] = res[(r["id"], x, k)]
            l0, l1 = lm.k4(v["own"], v["reask"]), lm.k4(own, re_)
            changed += l0 != l1
            if l0 not in ("skip", None): a.append((l0, bool(r["relied"])))
            if l1 not in ("skip", None): b.append((l1, bool(r["relied"])))
        s, t = score(a), score(b)
        print(f"{tag[5:]:24s} reported {s[0]:.3f} ({s[1]:.0%}/{s[2]:.0%}; {s[3]})  full-text {t[0]:.3f} "
              f"({t[1]:.0%}/{t[2]:.0%}; {t[3]})  {t[0] - s[0]:+.3f}")
    print("judgments flipped:", flips, "| influence labels changed:", changed)


if __name__ == "__main__":
    main()
