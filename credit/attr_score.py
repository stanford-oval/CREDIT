"""Teacher-forced attribution score (used for the upweighting runs of Section 6.1). Teacher-forces a model (base, or base + LoRA adapter) on
saved attribution transcripts up to and including the final 'RELIED ON:' and reads P(next token is 'none'); the
attribution score is 1 - P(none). Rows and labels: counterfactual influence from llm_adopt;
also records whether the answer gives the suggested answer (adopt, the agreeing-answer control).
Usage: python -m credit.attr_score MODEL OUT_JSON ADAPTERS(comma list, 'none' = base) ROOT:TAG:SPLIT [...]   (env LIMIT)"""
import json, math, os, sys
import numpy as np
import credit.llm_match as lm
from sklearn.metrics import roc_auc_score
from credit.common import MODELS, load_llm, sampling
from credit.gen import render
from credit.reliance import RELY

m, out_path, adapters = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
specs = [s.split(":") for s in sys.argv[4:]]
limit = int(os.environ.get("LIMIT", "0")) or None
cfg = MODELS[m]
use_lora = any(a != "none" for a in adapters)
llm = load_llm(cfg, max_model_len=12288, **(dict(enable_lora=True, max_lora_rank=64) if use_lora else {}))
tok = llm.get_tokenizer()
rows = []
for root, tag, split in specs:
    L = json.load(open(os.path.join(root, tag, f"llm_adopt_{split}.json")))
    for r in lm.items(root, tag, split, "syco"):
        k = lm.key(r)
        if k not in L or r.get("truncated"):
            continue
        lab = lm.k4(L[k]["own"], L[k]["reask"])
        if lab in ("skip", None):
            continue
        t = r["text"]
        if "</think>" not in t or "RELIED ON" not in t.split("</think>", 1)[1]:
            continue
        i = t.rindex("RELIED ON")
        j = t.index(":", i) + 1
        prompt = render(tok, [{"role": "user", "content": r["user"] + RELY}], cfg) + t[:j]
        rows.append(dict(q=r["id"], lab=bool(lab), adopt=bool(L[k]["own"]), stmt=bool(r["relied"]), kind=r["kind"], prompt=prompt))
rows = rows[:limit] if limit else rows
NONE = {i for s in ["none", " none", "None", " None"] for i in tok.encode(s, add_special_tokens=False)[:1]}
print(m, "rows", len(rows), "influential", sum(x["lab"] for x in rows), "agreeing", sum(x["adopt"] for x in rows),
      "none-token ids", sorted(NONE), flush=True)
res = {"model": m, "n": len(rows), "lab": [x["lab"] for x in rows], "adopt": [x["adopt"] for x in rows],
       "stmt": [x["stmt"] for x in rows], "q": [x["q"] for x in rows], "kind": [x["kind"] for x in rows], "scores": {}}
y = np.array(res["lab"]); a = np.array(res["adopt"])
for n_ad, ad in enumerate(adapters):
    lora = None
    if ad != "none":
        from vllm.lora.request import LoRARequest
        lora = LoRARequest(os.path.basename(ad.rstrip("/")), n_ad + 1, ad)
    outs = llm.generate([x["prompt"] for x in rows], sampling(max_tokens=1, temperature=0, top_p=1, logprobs=20), lora_request=lora)
    pn, mass = [], []
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        pn.append(sum(math.exp(v.logprob) for k, v in lp.items() if k in NONE))
        mass.append(sum(math.exp(v.logprob) for v in lp.values()))
    s = 1 - np.array(pn)
    res["scores"][ad] = s.tolist()
    print(f"[{os.path.basename(ad.rstrip('/'))}] AUROC all {roc_auc_score(y, s):.3f} | agreeing {roc_auc_score(y[a], s[a]):.3f} | "
          f"mean P(none) {np.mean(pn):.3f} | top-20 mass {np.mean(mass):.3f}", flush=True)
json.dump(res, open(out_path, "w"))
print("DONE", flush=True)
