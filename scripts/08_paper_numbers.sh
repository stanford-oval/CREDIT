#!/bin/bash
# Recompute the paper's numbers from the saved outputs and labels. CPU only; works directly on the released bundle
# (see data/README.md). Every step prints its results and saves them under $CREDIT_ROOT/numbers/ (<step>.txt, and .json
# where the module writes one). Not run here: the composition of the fresh set (credit.composition calls the LLM answer
# judge), and analyses whose outputs are not in the bundle (credit.infl_audit, the instruction-scope runs: scores in
# data/sysscope.json).
. "$(dirname "$0")/common.sh"
# Scoring needs the public sycophancy-eval file to tell TriviaQA from TruthfulQA questions.
SYCO="$CREDIT_ROOT/data/bench/sycophancy-eval"
[ -s "$SYCO/datasets/answer.jsonl" ] || git clone --depth 1 https://github.com/meg-tong/sycophancy-eval.git "$SYCO"
OUT="$CREDIT_ROOT/numbers"; mkdir -p "$OUT"
R8="$(results_for qwen3-8b)"; R14="$(results_for qwen3-14b)"; RA="$(results_for api)"
run() { local name="$1"; shift; log "$name"; "$@" | tee "$OUT/$name.txt"; }
FRONT="gpt-5-nano gpt-5-mini gpt-5 gpt-5.5 gpt-6.1-sol claude-haiku-4-5 claude-sonnet-5 claude-opus-5-5 gemini-3.5-flash-lite gemini-3.8-flash gemini-3.1-pro-preview deepseek-v4.1-flash"

# Main results, frontier models, transfer, baselines, ablations, DeepSeek-R1-Distill-Llama-8B
run paper_numbers $PY credit.paper_numbers --out "$OUT/paper_numbers.json"
run agree_conditional $PY credit.agree_conditional --out "$OUT/agree_conditional.json"   # Table: agreement
run by_kind $PY credit.by_kind --out "$OUT/by_kind.json"            # by suggestion correctness; Table: cases
run strict_prompt $PY credit.strict_prompt --out "$OUT/strict_prompt.json"
run label_checks $PY credit.label_checks --out "$OUT/label_checks.json"   # threshold sensitivity, label reliability
run selfknow $PY credit.selfknow_score "$HERE/data/self_knowledge" "$OUT/selfknow.json"

# Competing information: two pieces (Table: competing) and three pieces
for M in qwen3-8b qwen3-14b r1-qwen-7b; do R="$(results_for "$M")"; link_items "$R"
  for SP in test test2; do
    run "multi_${M}_$SP" $PY credit.multi_score --root "$R" --tags mb-base,mb-cfdiscL,mb-cfdisc4L,mb-cfmultiL --split "$SP" --json "$OUT/multi_${M}_$SP.json"
    run "multi3_${M}_$SP" $PY credit.multi_score --three --root "$R" --tags m3-base,m3-cfdisc4L --split "$SP" --json "$OUT/multi3_${M}_$SP.json"
  done
done

# Benchmark passages (FaithEval, RAMDocs)
run realdocs $PY credit.realdocs --root "$R8" --tags rely-base,rely-cfdisc4L,rely-cfdisc4L-s1,rely-cfdisc4L-s2,rely-cfdiscL,rely-cheah,rely-vftL,rely-vft

# Acknowledgment in the reasoning (VFT rows of Table: main; VFT with the attribution statement excluded from the loss)
run vft_reasoning env CREDIT_RESULTS="$R8" $PY credit.vft_cot --tags rely-base,rely-vftL,rely-vftuseL,rely-vftmaskL,rely-cfdisc4L --split test --llm

# Measurement checks: LLM answer judge vs string matching, keyword rule instead of the attribution judge, judge-free
# (lenient) rescoring
for R in "$R8" "$R14"; do
  $PY credit.llm_match --stage score --root "$R" --tags rely-base,rely-cfdisc4L --benches syco
  $PY credit.llm_match --stage score --root "$R" --tags apiq2-base,apiq2-cfdisc4L --benches syco --all_sources
done | tee "$OUT/answer_matching.txt"
run keyword_rule $PY credit.robust --root "$R8" --tags rely-base,rely-intro,rely-vftL,rely-vftuseL,rely-cfshufL,rely-cfdisc4L-s1,rely-cfdisc4L-s2,rely-cfdisc4L,rely-cheah --benches syco --split test
for R in "$R8" "$R14"; do $PY credit.formatblind --root "$R" --pairs rely-base:rely-cfdisc4L,x-base:x-cfdisc4L; done | tee "$OUT/judge_free.txt"

# Question source (TriviaQA vs TruthfulQA) on the large set, frontier models and CREDIT
for M in qwen3-8b qwen3-14b r1-qwen-7b; do $PY credit.breakdown --root "$(results_for "$M")" --tag apiq2-cfdisc4L --all_sources; done > "$OUT/breakdown.txt"
for m in $FRONT; do $PY credit.breakdown --root "$RA" --tag "api2-$m" --all_sources; done >> "$OUT/breakdown.txt"
log "breakdown"; cat "$OUT/breakdown.txt"

# Human annotations: judge validation and failure analysis
run judge_validation python "$HERE/data/annotations/judge_validation/score.py"
run failure_categories python "$HERE/data/annotations/failure_analysis/summarize.py"
run failure_stats python "$HERE/data/annotations/failure_analysis/failure_stats.py" --root "$CREDIT_ROOT"
log "wrote $OUT"
