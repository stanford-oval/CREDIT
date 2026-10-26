#!/bin/bash
# Baselines for one model (paper: Qwen3-8B). Each trained baseline then goes through 03_train.sh and 04_evaluate.sh.
. "$(dirname "$0")/common.sh"
M="${1:-qwen3-8b}"; R="$(results_for "$M")"; export CREDIT_RESULTS="$R" CF_ADOPT=llm
# VFT-style SFT: influence labels trained into the reasoning (at the start, or where the suggestion is first used).
# As run for the paper: user suggestions only, labels from one counterfactual answer (--label_rule flip1).
$PY credit.cf_disclose --stage build --model "$M" --label_rule flip1 --target cot --out_name sft_vftL.jsonl
$PY credit.cf_disclose --stage build --model "$M" --label_rule flip1 --target cot_use --out_name sft_vftuseL.jsonl
# VFT with the final attribution statement excluded from the loss (appendix control; eval tags rely-vftmaskL, t2-vftmaskL):
#   scripts/03_train.sh $M sft_vftL.jsonl ${M}_vftmaskL 42 --mask_from "RELIED ON:"
# Shuffled-label control
$PY credit.shuffle_labels --src sft_cf_llm.jsonl --out sft_cf_shufL.jsonl
# Faithfulness RL (GRPO, reward from one counterfactual sample)
# Reported run: Cheah et al. reward (binary, one re-ask), credit on all completion tokens, all 11,650 pooled prompts
$PY credit.rl_rely --stage pool
$PY credit.rl_rely --stage train --reward cheah --full_credit --pool_file pool.jsonl --out "$CREDIT_ROOT/runs/${M}_cheah" --max_steps 400 --save_steps 50
# Ablations (appendix: what drives the gain); train each with 03_train.sh and evaluate with 04_evaluate.sh
#   labels from one counterfactual answer instead of four (eval tag rely-cfflip1L; as run: user suggestions only):
$PY credit.cf_disclose --stage build --model "$M" --label_rule flip1 --target answer --out_name sft_cfflip1L.jsonl
#   scripts/03_train.sh $M sft_cfflip1L.jsonl ${M}_cfflip1L
#   labels from string matching instead of the LLM answer judge (eval tag rely-cfdisc4):
CF_ADOPT= $PY credit.cf_disclose --stage build --model "$M" --with_passages --with_channels --out_name sft_cf4.jsonl
#   loss only after the reasoning (the final answer and its attribution statement), or only on the attribution statement.
#   As run for the appendix ablation (eval tags rely-cfans on the test set, rely-cfline on the validation set): user
#   suggestions only, string-matched labels (sft_cf.jsonl):
CF_ADOPT= $PY credit.cf_disclose --stage build --model "$M" --out_name sft_cf.jsonl
#   scripts/03_train.sh $M sft_cf.jsonl ${M}_cfans 42 --loss_from "</think>"
#   scripts/03_train.sh $M sft_cf.jsonl ${M}_cfline 42 --loss_from "RELIED ON:"
#   The answer-section runs of Section 6.1 (eval tags rely-ans / t2-ans and rely-ans_s1 / t2-ans_s1) use CREDIT's data:
#   scripts/03_train.sh $M sft_cf4_llm.jsonl ${M}_cfdisc4L_obj_answer 42 --loss_from "</think>"
#   scripts/03_train.sh $M sft_cf4_llm.jsonl ${M}_cfdisc4L_obj_answer_s1 1 --loss_from "</think>"
#   Upweighting the agreeing, non-influential cases with a correct suggestion by duplication (x2 and x3; Section 6.1):
for W in 2 3; do $PY credit.upweight sft_cf4_llm.jsonl sft_cf4_llm_w$W.jsonl $W; done
#   scripts/03_train.sh $M sft_cf4_llm_w2.jsonl ${M}_cfdisc4L_w2;  scripts/03_train.sh $M sft_cf4_llm_w3.jsonl ${M}_cfdisc4L_w3
# Prompting baselines need no training:
#   scripts/04_evaluate.sh $M intro "" intro     (counterfactual question in the instruction)
# Stricter instruction (paper appendix): generate with --variant strict on the test, fresh, and large sets, then
#   python -m credit.strict_prompt --label   (re-asks are reused from the untrained run; labels via the LLM judge)
