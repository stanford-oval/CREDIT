#!/bin/bash
# Frontier models through the API on the large set (user suggestions); with NOTES=1 also on the system-prompt notes
# (answer key and neutral note, 105 questions; the first block of the frontier table).
# Usage: [NOTES=1] scripts/06_frontier.sh MODEL [MODEL ...]   (model names as your OpenAI-compatible endpoint exposes them)
. "$(dirname "$0")/common.sh"
R="$(results_for api)"; export CREDIT_RESULTS="$R"; link_items "$R"
for m in "$@"; do
  $PY credit.api_eval --model "$m" --tag "api2-$m" --items items_apiq2 --benches syco --split test --workers 10
  $PY credit.reliance --stage judge --tag "api2-$m" --split test
  $PY credit.llm_match --stage match --root "$R" --tags "api2-$m" --benches syco --all_sources
  if [ "${NOTES:-0}" = 1 ]; then
    $PY credit.api_eval --model "$m" --tag "api-$m" --items items_apiall --benches syco,x_sys,x_sysn --split test --workers 10
    $PY credit.reliance --stage judge --tag "api-$m" --split test
    $PY credit.llm_match --stage match --root "$R" --tags "api-$m" --benches syco,x_sys,x_sysn
  fi
done
