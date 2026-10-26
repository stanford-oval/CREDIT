"""Instruction-scope control (Appendix: transfer and system-prompt notes): the attribution instruction names the system
prompt explicitly (`reliance --variant sysscope`). Qwen3-8B, untrained and CREDIT, on the held-out types (test and fresh
sets). Re-asks are reused from the standard runs (x-base, x-cfdisc4L, t2x-cfdisc4L), so the influence labels are shared
and every comparison is paired. Copies the re-asks, runs the LLM answer judge (API) on the new runs, and writes the
scores to $CREDIT_ROOT/results_api/sysscope.json (the paper's scores are in data/sysscope.json).
Usage: python -m credit.sysscope_prompt
"""
import json
import os
import subprocess

from credit.common import read_jsonl, write_jsonl
import credit.paper_numbers as pn

A = os.environ.get("CREDIT_ROOT", "."); R = f"{A}/results"; XB = "x_tool,x_peer,x_meta,x_sys,x_cred"
RUNS = [("untrained", "x-sysscope-base", {"test": "x-base", "test2": "x-base"}),
        ("CREDIT", "x-sysscope-cfdisc4L", {"test": "x-cfdisc4L", "test2": "t2x-cfdisc4L"})]


def main():
    out = {}
    for name, tag, base in RUNS:
        for sp in ("test", "test2"):
            p = f"{R}/{tag}/reliance_{sp}.jsonl"
            if not os.path.exists(p): print(name, sp, "missing"); continue
            ref = pn.resolve(R, base[sp], sp)  # fresh-set runs may be named x-* or t2x-*
            B = {(r["bench"], r["id"], r["kind"]): r for r in read_jsonl(f"{R}/{ref}/reliance_{sp}.jsonl")}
            Rr = read_jsonl(p)
            for r in Rr:
                b = B.get((r["bench"], r["id"], r["kind"]))
                if b and r["bench"].startswith("x_"):
                    for k in ("reask", "reask_key", "reask_keys_k"): r[k] = b.get(k)
            write_jsonl(p, Rr)
            subprocess.run(f"CREDIT_RESULTS={R} python -u -m credit.llm_match --stage match --root {R} --tags {tag} --benches {XB} --split {sp}",
                           shell=True, check=True, stdout=subprocess.DEVNULL)
            for b in XB.split(","):
                S = pn.rows(R, tag, sp, b); T = pn.rows(R, ref, sp, b)
                if not S or not T: continue
                s, t = pn.score(S), pn.score(T); d = pn.paired(S, T)
                out[f"{name}|{sp}|{b}"] = dict(scope=round(s[0], 3), attr=[round(s[1], 3), round(s[2], 3)], standard=round(t[0], 3), diff=d, n=[s[3], s[4]])
                print(f"{name:9s} {sp:5s} {b:7s} system-in-scope {s[0]:.3f} ({s[1]:.0%}/{s[2]:.0%}; n {s[3]}/{s[4]}) | standard {t[0]:.3f} | diff {d[0]:+.3f} [{d[1]:+.2f}, {d[2]:+.2f}]", flush=True)
    json.dump(out, open(f"{A}/results_api/sysscope.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
