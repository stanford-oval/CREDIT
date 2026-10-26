# Released data

## In this repository

`annotations/judge_validation/`: two annotators independently labeled a stratified sample of 87 self-attribution
statements (task 1) and 100 final answers (task 2) without seeing the judges' decisions. `score.py` reproduces the
agreement numbers in the paper (judge-annotator kappa .95/.89 and .78/.98; inter-annotator .84 and .76; the judges match
every item on which the annotators agree).

`annotations/failure_analysis/`: two annotators categorized 100 test errors of Qwen3-8B with CREDIT (50 omissions, 50
false attributions), 50 errors each. `summarize.py` reproduces the category counts. `failure_stats.py` reproduces the
other numbers of the failure-analysis appendix: counts by type of external information, correctness of the
information, re-ask counts, and the attribution statements drafted in the reasoning; with `--root $CREDIT_ROOT` it also
gives the numbers over all scored responses (error totals, error rates by re-ask count, draft versus final statement).
Annotator identities are replaced by `annotator_1` and `annotator_2` (numbered separately for each task).

## Data bundle (separate storage)

The larger files are gzipped tar archives that unpack directly into `$CREDIT_ROOT`, in the layout the code reads
(`results/`, `results_q14/`, `results_r1q/`, `results_r1l/`, `results_api/`). The archives needed to recompute the
paper's numbers are hosted on OSF: https://osf.io/zm5ae/. `scripts/download_bundle.sh` checks them against
`data/SHA256SUMS`.

| Archive | Contents | Size |
|---|---|---|
| `credit_outputs.tar.gz` | all model outputs behind the paper's numbers, for Qwen3-8B, Qwen3-14B, DeepSeek-R1-Distill-Qwen-7B, 11 frontier models, and one large open model (DeepSeek-V4.1-Flash): responses, `RELIED ON` lines and judge decisions, re-asked answers, and LLM-judged influence labels (`llm_adopt_<split>.json`) | 144 MB |
| `credit_outputs_extra.tar.gz` | outputs added after the first release: VFT with the attribution statement excluded from the loss (`results/rely-vftmaskL`, `results/t2-vftmaskL`) and DeepSeek-R1-Distill-Llama-8B, untrained and CREDIT, on the test, fresh, large, and transfer sets (`results_r1l/`) | 14 MB |
| `credit_items.tar.gz` | every evaluation item file: validation, test, fresh, large sets; held-out types (templates A, B, neutral); competing information (two and three pieces) | 2.5 MB |
| `credit_summaries.tar.gz` | per-run summary scores (JSON) | 32 KB |

```bash
bash scripts/download_bundle.sh      # downloads from OSF, verifies checksums, unpacks into $CREDIT_ROOT
bash scripts/08_paper_numbers.sh     # recomputes the paper's numbers from the outputs into $CREDIT_ROOT/numbers/
```

Three further archives hold the training data and the trained models: `credit_training.tar.gz` (537 MB; training
question pools, training generations and re-asks, influence labels, and the relabeled SFT sets for every training
condition), `credit_adapters.tar.gz` (2.0 GB; LoRA adapters of CREDIT for Qwen3-8B, Qwen3-14B, and
DeepSeek-R1-Distill-Qwen-7B), and `credit_r1-llama-8b.tar.gz` (677 MB; the CREDIT adapter and training data of
DeepSeek-R1-Distill-Llama-8B). Fetch them with
`ARCHIVES="credit_training.tar.gz credit_adapters.tar.gz credit_r1-llama-8b.tar.gz" bash scripts/download_bundle.sh`.

Per-run files: `<root>/<tag>/reliance_<split>.jsonl` holds one record per item (`bench`, `id`, `kind` = wrong or correct
suggestion, the response `text`, `rely_line`, judge decisions `relied` / `relied_src`, the extracted answer `own_key`,
and the re-asked answers `reask_keys_k`). `<root>/<tag>/llm_adopt_<split>.json` maps `bench|id|kind` to the LLM answer
judge's decisions for the answer (`own`) and each re-ask (`reask`). Run tags are explained in `docs/PIPELINE.md`.

## Licenses and terms

The data is derived from sycophancy-eval (TriviaQA, TruthfulQA), MMLU, ARC, CommonsenseQA, FaithEval, and RAMDocs,
which remain under their own licenses, and contains outputs of open models (Qwen3, DeepSeek-R1-Distill) and commercial
API models (OpenAI, Anthropic, Google, DeepSeek), released for research on model behavior and subject to the providers'
terms. See `NOTICE`.
