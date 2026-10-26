# Pipeline and paper map

## Stages

| Stage | Script | Modules | Output |
|---|---|---|---|
| Public benchmarks | `00_fetch_data.sh` | `fetch_data` | `$CREDIT_ROOT/data/bench/` |
| Evaluation items | `01_build_eval_sets.sh` | `pub_eval`, `build_test2`, `build_xfer`, `build_multi`, `build_multi3`, `api_items` | `results*/pub_eval/items*.jsonl` |
| Training questions | `02_build_training_data.sh` | `rl_check` (TriviaQA), `factor_data` (MMLU), `rsrc_data` + `passages` (passages) | `rl_check/`, `factor/`, `rsrc/`, `passages/` |
| Generate + counterfactual answers | `02_build_training_data.sh` | `cf_disclose --stage gen`, `factor_data --stage closed` | `cf_disclose/gen*.jsonl`, `closed*.jsonl` |
| Label + relabel | `02_build_training_data.sh` | `cf_disclose --stage llm_label/build`; with `MULTI=1` also `cf_multi` (two pieces) | `cf_disclose/sft_cf4_llm.jsonl`, `cf_multi/sft_cf4multi.jsonl` |
| Fine-tune | `03_train.sh` | `sft` (LoRA rank 64, one epoch) | `runs/<model>_cfdisc4L/` |
| Evaluate | `04_evaluate.sh` (`VARIANTS=1`: appendix variants) | `reliance` (gen, judge, judge_src, reask), `llm_match` | `<run>/reliance_<split>.jsonl`, `llm_adopt_<split>.json` |
| Competing information | `05_competing.sh` | `reliance`, `llm_match` | `mb-*`, `m3-*` runs |
| Frontier models | `06_frontier.sh` (`NOTES=1`: system-prompt notes) | `api_eval`, `reliance --stage judge`, `llm_match` | `results_api/api2-*`, `results_api/api-*` |
| Baselines and ablations | `07_baselines.sh` | `cf_disclose --target cot/cot_use` (VFT), `shuffle_labels`, `rl_rely` (faithfulness RL), `cf_disclose --label_rule flip1`, `sft --loss_from`, `reliance --variant intro/strict` | runs `rely-vftL`, `rely-cfshufL`, `rely-cheah`, `rely-cfflip1L`, `rely-intro`, `rely-strict` |
| Numbers | `08_paper_numbers.sh` | see the paper map | `$CREDIT_ROOT/numbers/` |

Influence labels (Definition 1): for each piece `s`, the answer `a ~ M(q, S)` and N=4 counterfactual answers
`a'_i ~ M(q, S \ {s})`; with `c = sum_i match(a'_i, s)`, `s` is influential if `match(a, s) = 1` and `c <= L = 1`,
non-influential if `match(a, s) = 0` or `c >= U = 3`, and unscored otherwise (`llm_match.k4`, which compares the share
`c / n` of the `n >= 3` valid counterfactual answers with 1/4 and 3/4). `match` is the LLM answer judge (`llm_match`);
the attribution judge reads the RELIED ON line (`reliance --stage judge`; `judge_src` also requires the piece's source).

## Paper map

The last column names the module, and the file in `$CREDIT_ROOT/numbers/` that `08_paper_numbers.sh` writes.

| Paper | Runs (tags) | Module: output |
|---|---|---|
| Frontier models (Figure 1, Table: frontier) | `results_api/api2-*` (large set), `results_api/api-*` (system-prompt notes) | `paper_numbers`: `paper_numbers` |
| Main results (Table: main) | `rely-*` (test) | `paper_numbers`: `paper_numbers`; reasoning rows `vft_cot --llm`: `vft_reasoning` |
| Fresh and large sets (Table: agreement) | `rely-*`, `t2-*` (fresh), `apiq2-*` (large) | `paper_numbers`, `agree_conditional`: `paper_numbers`, `agree_conditional` |
| By suggestion correctness; case types (Appendix: agreeing answers, Table: cases) | as above; frontier `api2-*` | `by_kind`: `by_kind` |
| A Llama-based model (Appendix) | `results_r1l/` | `paper_numbers`, `agree_conditional` |
| What drives the gain (Appendix) | `rely-cfdisc4` (string labels), `rely-cfflip1L` (one counterfactual answer), `rely-vftmaskL`, `t2-vftmaskL` | `paper_numbers`, `vft_cot --llm`: `paper_numbers`, `vft_reasoning` |
| Transfer across types (Table: transfer and appendix) | `x-*` (template A), `xb-*` (template B), `xc-*` (neutral note), `t2x-*` (fresh set), `x-cfdisc4L-s1/-s2` | `paper_numbers`: `paper_numbers` |
| Prompting baselines (Appendix: stricter instruction) | `rely-strict`, `apiq2-strict` (counterfactual answers from `rely-base`/`t2-base`/`apiq2-base`) | `reliance --variant strict`, `strict_prompt`: `strict_prompt` |
| Instruction scope (Appendix: system-prompt notes) | `x-sysscope-base`, `x-sysscope-cfdisc4L` (counterfactual answers from `x-*`, `t2x-*`) | `reliance --variant sysscope`, `sysscope_prompt`; scores in `data/sysscope.json` |
| Asking the model whether it knows (Appendix) | P(IK) scores in `data/self_knowledge/` | `pik` (GPU); `selfknow_score`: `selfknow` |
| Benchmark passages (FaithEval, RAMDocs) | `rely-*` (faitheval, ramdocs benches) | `realdocs`: `realdocs` |
| Competing information (Table: competing; three pieces) | `mb-*` (two pieces), `m3-*` (three pieces), `mb-cfmultiL` | `multi_score`: `multi_<model>_<split>`, `multi3_<model>_<split>` |
| Composition of the questions (Appendix) | `rely-base`, `t2-*` | `composition` (calls the LLM answer judge; not run by `08`) |
| Failure analysis (Appendix) | `rely-cfdisc4L`, `t2-cfdisc4L`, `x-cfdisc4L` (and `rely-base`, `x-base` for the untrained drafts) | `data/annotations/failure_analysis/`: `failure_categories`, `failure_stats` |
| Measurement checks (Appendix) | all | answer matching `llm_match --stage score`: `answer_matching`; keyword rule `robust`: `keyword_rule`; judge-free rescoring `formatblind`: `judge_free`; thresholds and label reliability `label_checks`: `label_checks`; question source `breakdown`: `breakdown`; label audit `infl_audit` (GPU; outputs not released) |
| Judge validation (Appendix) | stratified samples | `data/annotations/judge_validation/score.py`: `judge_validation` |
| Loss on the answer section; upweighting (Section: what drives the gain; Appendix table: cases) | `rely-ans`, `t2-ans`, `rely-ans_s1`, `t2-ans_s1` (answer section, seeds 42 and 1); upweighting x2/x3 (`runs/qwen3-8b_cfdisc4L_w2`, `_w3`; no generated runs) | answer section: `07_baselines.sh` commands, scored like the main runs; upweighting: `upweight`, then `attr_score qwen3-8b OUT none,<CREDIT adapter>,<upweighted adapter> $CREDIT_RESULTS:rely-base:test $CREDIT_RESULTS:rely-base:test2` (teacher-forced on the untrained transcripts) |
| Predictability from the question (Appendix: measurement checks) | `rely-cfdisc` (first CREDIT run, validation set) | `proxy_deploy --tag rely-cfdisc`: `rely-cfdisc/proxy_deploy.json` (`deployable` block) |
| Full-text answer matching of long frontier answers (Appendix: frontier settings) | `results_api/api2-*` | `rejudge_full` (calls the LLM answer judge) |
| Figures 1-3 | - | TikZ sources in `figures/` (`pdflatex`) |

Run tags: `<set>-<model variant>`. Sets: `rely` = test set, user suggestions; `x`/`xb`/`xc` = held-out types (templates
A, B, neutral note); `t2` = fresh set; `t2x` = held-out types on the fresh set; `apiq2` = large set; `mb`/`m3` =
two/three competing pieces; `api2-<name>` = frontier model on the large set; `api-<name>` = frontier model on the 105
questions with system-prompt notes. Some fresh-set runs are stored under the test-set tag (`rely-base`, `x-base`, split
`test2`); the scorers accept either name. Variants: `base` = untrained; `cfdisc4L` = CREDIT (four training types,
LLM-judged labels); `cfdiscL` = suggestions only; `cfdisc2L` = suggestions + passages; `cfmultiL` = + two-piece
training; `-s1`/`-s2` = extra seeds; `cfdisc4` = string-matched labels; `cfflip1L` = labels from one counterfactual
answer; `intro` = counterfactual question in the instruction; `strict` = stricter instruction; `vftL`/`vftuseL` =
VFT-style SFT; `vftmaskL` = VFT with the attribution statement excluded from the loss; `cfshufL` = shuffled labels;
`cheah` = faithfulness RL.

Further options (not part of the CREDIT recipe): `sft --focus_tokens` (with `--focus_last`: upweight the first or last
N completion tokens), `cf_disclose --target both/both_use/selfpred` and `--include_ambiguous`, `rl_rely --reward ours`,
and the model keys in `credit/common.py` other than the four above.
