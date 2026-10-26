#!/bin/bash
# Build CREDIT training data for one model: generate answers with the self-attribution instruction, re-ask without
# the external information, label influence with the LLM answer judge, and relabel the RELIED ON line.
# Usage: [MULTI=1] scripts/02_build_training_data.sh qwen3-8b|qwen3-14b|r1-qwen-7b|r1-llama-8b
# MULTI=1 also builds the two-piece training data ("+ multi" in the competing-information table).
. "$(dirname "$0")/common.sh"
M="$1"; R="$(results_for "$M")"; Q="$(results_for qwen3-8b)"; export CREDIT_RESULTS="$R"
mkdir -p "$R"/{rl_check,factor,rsrc,passages,cf_disclose}
if [ "$R" = "$Q" ]; then   # shared question pools: built once (the API writes wrong suggestions and passages)
  [ -s "$Q/rl_check/trivia.jsonl" ] || $PY credit.rl_check --stage trivia           # TriviaQA training questions
  [ -s "$Q/rsrc/claims.jsonl" ]     || $PY credit.rsrc_data --stage claims          # ARC/CSQA questions for passages
  [ -s "$Q/passages/passages.jsonl" ] || $PY credit.passages --stage gen            # generated passages (MMLU/ARC/CSQA)
else
  for f in rl_check/trivia.jsonl factor/pool_mmlu.jsonl rsrc/claims.jsonl passages/passages.jsonl; do
    ln -sfn "$Q/$f" "$R/$f"; done
fi
$PY credit.factor_data --stage closed --model "$M" --k 4                 # re-asks for the passage questions (ARC/CSQA)
$PY credit.factor_data --stage closed --pool mmlu --model "$M" --k 4     # re-asks for MMLU questions (also builds the MMLU pool)
$PY credit.cf_disclose --stage gen --model "$M"                          # user suggestions (+ TriviaQA re-asks)
$PY credit.cf_disclose --stage gen --model "$M" --sources passage        # retrieved passages
$PY credit.cf_disclose --stage gen --model "$M" --sources channels --n 1 # tool outputs and another AI's answers
$PY credit.cf_disclose --stage llm_label --model "$M" --with_passages --with_channels
export CF_ADOPT=llm
$PY credit.cf_disclose --stage stats --model "$M" --with_passages --with_channels
$PY credit.cf_disclose --stage build --model "$M" --with_passages --with_channels --out_name sft_cf4_llm.jsonl  # CREDIT
$PY credit.cf_disclose --stage build --model "$M" --out_name sft_cf_llm.jsonl                                 # suggestions only
$PY credit.cf_disclose --stage build --model "$M" --with_passages --out_name sft_cf2_llm.jsonl                # + passages
if [ "${MULTI:-0}" = 1 ]; then  # two pieces per prompt, labeled by removing each; mixed with the CREDIT data
  $PY credit.cf_multi --stage gen --model "$M"
  $PY credit.cf_multi --stage label --model "$M"
  $PY credit.cf_multi --stage build --model "$M"   # cf_multi/sft_cf4multi.jsonl -> 03_train.sh $M sft_cf4multi.jsonl ${M}_cfmultiL
fi
log "training data: $R/cf_disclose/sft_cf4_llm.jsonl"
