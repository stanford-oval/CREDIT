#!/bin/bash
# Shared setup for all scripts: load .env, choose the results directory for a model, and define helpers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$HERE/.env" ] && set -a && . "$HERE/.env" && set +a
export CREDIT_ROOT="${CREDIT_ROOT:-$HERE/workdir}"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CREDIT_ROOT"
PY="python -u -m"
XB=x_tool,x_peer,x_meta,x_sys,x_cred          # held-out-type benches (user suggestions are 'syco')
# Results layout used by the paper scripts: one directory per model family.
results_for() { case "$1" in
  qwen3-8b) echo "$CREDIT_ROOT/results";; qwen3-14b) echo "$CREDIT_ROOT/results_q14";;
  r1-qwen-7b) echo "$CREDIT_ROOT/results_r1q";; r1-llama-8b) echo "$CREDIT_ROOT/results_r1l";;
  api) echo "$CREDIT_ROOT/results_api";;
  *) echo "$CREDIT_ROOT/results_$1";; esac; }
# Evaluation item files live in the main results directory; other model directories link to them.
link_items() { local R="$1"; mkdir -p "$R/pub_eval"
  [ "$R" = "$CREDIT_ROOT/results" ] && return 0
  for f in "$CREDIT_ROOT"/results/pub_eval/items*.jsonl; do [ -e "$f" ] && ln -sfn "$f" "$R/pub_eval/$(basename "$f")"; done; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
