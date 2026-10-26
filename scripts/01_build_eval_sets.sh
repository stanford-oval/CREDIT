#!/bin/bash
# Build the evaluation item files. The released data bundle already contains all of them (credit_items.tar.gz); prefer it
# for exact reproduction. This script rebuilds the validation/test splits, the fresh set, the held-out types, and the
# competing-information prompts from the public benchmarks, and the frontier-model items, including the large set
# (items_apiq2_test.jsonl: the remaining 891 TriviaQA + 817 TruthfulQA questions), which it rebuilds exactly.
. "$(dirname "$0")/common.sh"
export CREDIT_RESULTS="$(results_for qwen3-8b)"
$PY credit.pub_eval --stage build --n 200                 # validation/test splits of sycophancy-eval, FaithEval, RAMDocs
$PY credit.pub_eval --stage build --split val2 --n 100    # second validation split
$PY credit.build_test2 --n 500 --seed 21                  # fresh set, drawn after the method was frozen
$PY credit.build_xfer                                     # held-out types, template A
$PY credit.build_xfer b                                   # template B (paraphrased)
$PY credit.build_xfer c                                   # neutral system-prompt note
$PY credit.build_multi                                    # two competing pieces
$PY credit.build_multi3                                   # three competing pieces (an LLM writes the second wrong answer)
# Frontier models: 105 questions with user suggestions and system-prompt notes, then the large set (every remaining
# question; the open models are evaluated on it too)
CREDIT_RESULTS="$(results_for api)" $PY credit.api_items --n 105 --seed 11
CREDIT_RESULTS="$(results_for api)" $PY credit.api_items --dataset all --n 0 --exclude items_apiall --prefix items_apiq2 --seed 12
cp "$(results_for api)/pub_eval/items_apiq2_test.jsonl" "$CREDIT_RESULTS/pub_eval/"
