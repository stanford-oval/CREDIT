#!/bin/bash
# End-to-end reproduction for one open model (GPU). Frontier models and baselines are separate (06, 07).
. "$(dirname "$0")/common.sh"
M="${1:-qwen3-8b}"; D="$(dirname "$0")"
bash "$D/00_fetch_data.sh"
[ -s "$CREDIT_ROOT/results/pub_eval/items_test.jsonl" ] || bash "$D/01_build_eval_sets.sh"
bash "$D/02_build_training_data.sh" "$M"
bash "$D/03_train.sh" "$M"
bash "$D/04_evaluate.sh" "$M" base
bash "$D/04_evaluate.sh" "$M" cfdisc4L "$CREDIT_ROOT/runs/${M}_cfdisc4L"
bash "$D/05_competing.sh" "$M" base
bash "$D/05_competing.sh" "$M" cfdisc4L "$CREDIT_ROOT/runs/${M}_cfdisc4L"
bash "$D/08_paper_numbers.sh"
