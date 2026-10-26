#!/bin/bash
# Competing external information: two pieces (mb-*) and three pieces (m3-*) per prompt, scored piece by piece.
# Usage: scripts/05_competing.sh MODEL NAME [ADAPTER_DIR]
. "$(dirname "$0")/common.sh"
M="$1"; NAME="$2"; AD="${3:+--adapter $3}"; R="$(results_for "$M")"; export CREDIT_RESULTS="$R"; link_items "$R"
for SET in "mb items_multi m_sugg,m_tool,m_peer,m_meta" "m3 items_multi3 $(echo m3_{sugg,tool,peer}{0,1,2} | tr ' ' ,)"; do
  set -- $SET; P="$1"; ITEMS="$2"; B="$3"; T="$P-$NAME"
  $PY credit.reliance --stage gen --model "$M" --tag "$T" $AD --variant rely --items "$ITEMS" --split test,test2
  $PY credit.reliance --stage judge --tag "$T" --split test,test2
  $PY credit.reliance --stage judge_src --tag "$T" --split test,test2
  $PY credit.reliance --stage reask --model "$M" --tag "$T" $AD --items "$ITEMS" --split test,test2
  for SP in test test2; do $PY credit.llm_match --stage match --root "$R" --tags "$T" --benches "$B" --split "$SP"; done
done
