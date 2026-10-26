#!/bin/bash
# LoRA fine-tuning (rank 64, one epoch, effective batch 16). Usage:
#   scripts/03_train.sh MODEL [DATA_NAME=sft_cf4_llm.jsonl] [OUT_NAME=MODEL_cfdisc4L] [SEED=42] [extra credit.sft flags]
#   e.g. the extra seeds: scripts/03_train.sh qwen3-8b sft_cf4_llm.jsonl qwen3-8b_cfdisc4L-s1 1
. "$(dirname "$0")/common.sh"
M="$1"; DATA="${2:-sft_cf4_llm.jsonl}"; OUT="${3:-${M}_cfdisc4L}"; SEED="${4:-42}"
R="$(results_for "$M")"; export CREDIT_RESULTS="$R"; mkdir -p "$CREDIT_ROOT/runs"
DIR=cf_disclose; [[ "$DATA" == *multi* ]] && DIR=cf_multi
$PY credit.sft --model "$M" --data "$R/$DIR/$DATA" --out "$CREDIT_ROOT/runs/$OUT" --epochs 1 --bsz 16 --seed "$SEED" "${@:5}"
