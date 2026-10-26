#!/bin/bash
# Evaluate one model (untrained or with an adapter) on the test set, held-out types, fresh set, and large set:
# generate with the instruction, judge the RELIED ON line, re-ask without the external information (k=4), and label
# influence with the LLM answer judge. Usage:
#   [VARIANTS=1] scripts/04_evaluate.sh MODEL NAME [ADAPTER_DIR] [VARIANT=rely|intro|strict]
#   VARIANTS=1 adds the appendix variants: template B, the neutral system-prompt note, the fresh-set transfer, and the
#   instruction-scope variant (scored by credit.sysscope_prompt).
#   e.g. scripts/04_evaluate.sh qwen3-8b base ; scripts/04_evaluate.sh qwen3-8b cfdisc4L workdir/runs/qwen3-8b_cfdisc4L
. "$(dirname "$0")/common.sh"
M="$1"; NAME="$2"; AD="${3:+--adapter $3}"; VAR="${4:-rely}"
R="$(results_for "$M")"; export CREDIT_RESULTS="$R"; link_items "$R"
ev() {  # tag items split
  $PY credit.reliance --stage gen --model "$M" --tag "$1" $AD --variant "$VAR" --items "$2" --split "$3"
  $PY credit.reliance --stage judge --tag "$1" --split "$3"
  $PY credit.reliance --stage reask --model "$M" --tag "$1" $AD --items "$2" --split "$3"; }
ev "rely-$NAME" items test;        $PY credit.llm_match --stage match --root "$R" --tags "rely-$NAME" --benches syco
ev "x-$NAME" items_xfer test
$PY credit.reliance --stage judge_src --tag "x-$NAME" --split test
$PY credit.llm_match --stage match --root "$R" --tags "x-$NAME" --benches "$XB"
ev "t2-$NAME" items test2;         $PY credit.llm_match --stage match --root "$R" --tags "t2-$NAME" --benches syco --split test2
ev "apiq2-$NAME" items_apiq2 test; $PY credit.llm_match --stage match --root "$R" --tags "apiq2-$NAME" --benches syco --all_sources
$PY credit.llm_match --stage score --root "$R" --tags "rely-$NAME" --benches syco
if [ "${VARIANTS:-0}" = 1 ]; then
  ev "xb-$NAME" items_xferb test;   $PY credit.reliance --stage judge_src --tag "xb-$NAME" --split test     # template B
  $PY credit.llm_match --stage match --root "$R" --tags "xb-$NAME" --benches "$XB"
  ev "xc-$NAME" items_xferc test;   $PY credit.llm_match --stage match --root "$R" --tags "xc-$NAME" --benches x_sys  # neutral note
  ev "t2x-$NAME" items_xfer test2;  $PY credit.reliance --stage judge_src --tag "t2x-$NAME" --split test2  # fresh set
  $PY credit.llm_match --stage match --root "$R" --tags "t2x-$NAME" --benches "$XB" --split test2
  # instruction scope: credit.sysscope_prompt reuses the counterfactual answers of x-$NAME / t2x-$NAME
  $PY credit.reliance --stage gen --model "$M" --tag "x-sysscope-$NAME" $AD --variant sysscope --items items_xfer --split test,test2
  $PY credit.reliance --stage judge --tag "x-sysscope-$NAME" --split test,test2
  $PY credit.reliance --stage judge_src --tag "x-sysscope-$NAME" --split test,test2
fi
