"""Input-only predictors of influence vs. the model's own disclosure (external review: the earlier predictor used gold
suggestion correctness, which a deployed system lacks).

Population: validation sycophancy answers of --tag that ADOPT the suggestion, k=4 labels (ambiguous excluded).
Train: the relabeling training generations (TriviaQA + MMLU suggestions) that adopt, with their training labels.
Predictors (TF-IDF word 1-2 grams + logistic regression):
  oracle     : question + KIND_<true|false> token (the earlier analysis; uses gold correctness)
  deployable : question + the suggested answer string (no gold correctness)
Fixes vs. the earlier scratch script: the FULL question text at test time (it used the 80-character item id), and a
bootstrap that resamples whole questions (both suggestion kinds of a question together).
Usage: python -m credit.proxy_deploy --tag rely-cfdisc
The paper's numbers (appendix "How much is predictable from the question and external information?": AUROC 0.78,
and 0.87 with the attribution statement) are the `deployable` block of $CREDIT_RESULTS/rely-cfdisc/proxy_deploy.json:
the first CREDIT run (user suggestions, string-matched labels) on the validation set.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from .common import read_jsonl, results_dir
from .cf_disclose import suggestion_rows, closed_adoption, adopts, influence
from .infl_audit import k4_label, adopt_key
from .rl_check import claim_answer, name_matchable


def question_of(user, src):
    sep = " I think the answer is " if src == "trivia" else "\n\nI think the answer is"
    return user.split(sep)[0]


def ba(y, p):
    return 0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rely-cfdisc")
    ap.add_argument("--split", default="val,val2")
    ap.add_argument("--B", type=int, default=500)
    args = ap.parse_args()
    root = results_dir()
    rows = suggestion_rows(root)
    cba = closed_adoption(root, rows)
    tr = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(root, "cf_disclose", "gen_[0-9]*.jsonl"))):
        for g in read_jsonl(p):
            if g["finish"] == "length" or not adopts(g["text"], g):
                continue
            i = influence(True, cba.get((g["src"], g["qid"], g["kind"])))
            if i is None:
                continue
            q = question_of(g["user"], g["src"])
            tr["oracle"].append(f"KIND_{g['kind']} SRC_{g['src']} {q}")
            tr["deployable"].append(f"SRC_{g['src']} {q} SUGGESTED {g['sugg']}")
            tr["y"].append(int(i))
    ytr = np.array(tr["y"])
    te = defaultdict(list)
    for sp in args.split.split(","):
        for r in read_jsonl(os.path.join(root, args.tag, f"reliance_{sp}.jsonl")):
            if (r["bench"] != "syco" or not name_matchable(r) or r.get("relied") is None or r.get("reask_key") is None
                    or r.get("truncated", "</think>" not in r["text"]) or not adopt_key(r["own_key"], r)):
                continue
            lab = k4_label(r)
            if lab is None:
                continue
            q = question_of(r["user"], "trivia")
            te["id_prefix_mismatch"].append(int(not q.startswith(r["id"])))  # sanity: the id is the question's prefix
            te["oracle"].append(f"KIND_{r['kind']} SRC_trivia {q}")
            te["deployable"].append(f"SRC_trivia {q} SUGGESTED {claim_answer(r['claim'])}")
            te["y"].append(int(lab)); te["dis"].append(int(bool(r["relied"]))); te["qid"].append(r["id"])
    y, dis, qid = np.array(te["y"]), np.array(te["dis"]), np.array(te["qid"])
    out = dict(tag=args.tag, train_adopted=len(ytr), train_influenced_share=round(float(ytr.mean()), 3), test_adopted=len(y),
               test_influenced=int(y.sum()), model_disclosure_ba=round(float(ba(y, dis)), 3),
               id_prefix_mismatch=int(sum(te["id_prefix_mismatch"])))
    print(f"[proxy] train {len(ytr)} (influenced {ytr.mean():.3f}) | test {len(y)} (influenced {y.sum()}) | model BA {ba(y, dis):.3f}", flush=True)
    uq = np.unique(qid)
    rng = np.random.default_rng(0)
    for name in ("oracle", "deployable"):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(vec.fit_transform(tr[name]), ytr)
        s = np.clip(clf.predict_proba(vec.transform(te[name]))[:, 1], 1e-6, 1 - 1e-6)
        Z1 = np.c_[np.log(s / (1 - s))]
        Z2 = np.c_[np.log(s / (1 - s)), dis]
        cv = lambda Z, yy: cross_val_predict(LogisticRegression(max_iter=1000), Z, yy, cv=5, method="predict_proba")[:, 1]
        a1, a2 = roc_auc_score(y, cv(Z1, y)), roc_auc_score(y, cv(Z2, y))
        gains = []
        for _ in range(args.B):  # resample whole questions
            pick = rng.choice(uq, len(uq))
            idx = np.concatenate([np.flatnonzero(qid == q) for q in pick])
            yb = y[idx]
            if yb.min() == yb.max():
                continue
            try:
                gains.append(roc_auc_score(yb, cv(Z2[idx], yb)) - roc_auc_score(yb, cv(Z1[idx], yb)))
            except ValueError:
                continue
        g = np.array(gains)
        res = dict(auroc=round(float(roc_auc_score(y, s)), 3), ba_at_05=round(float(ba(y, (s > .5).astype(int))), 3),
                   cv_auroc_predictor=round(float(a1), 3), cv_auroc_predictor_plus_disclosure=round(float(a2), 3),
                   gain_mean=round(float(g.mean()), 3), gain_ci=[round(float(np.percentile(g, 2.5)), 3), round(float(np.percentile(g, 97.5)), 3)],
                   p_gain_le_0=round(float(np.mean(g <= 0)), 3), n_boot=len(g))
        out[name] = res
        print(f"[proxy] {name}: {res}", flush=True)
    json.dump(out, open(os.path.join(root, args.tag, "proxy_deploy.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
