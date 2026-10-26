# CREDIT: Counterfactual Relabeling for Dependence on Information Training

Code and data for **"Did the Model Use That? Counterfactual Training Boosts Self-Attribution in Language Models"**
by Zikai Zhou, Jiacheng Sang, Yilin Xu, Yufei Jin, and Monica S. Lam.

When a model's answer agrees with information a user supplied (their own suggestion, a retrieved document, a tool's
output), the user cannot tell from the answer whether the model confirmed it or merely echoed it. We call stating which
external information an answer depends on **self-attribution**, and measure it with a removal test: a piece of
external information is **influential** if the model gives the answer it supports with the information present but
rarely without it. **CREDIT** trains a model to state, in the answer itself, which external information was influential.
It generates answers, labels influence with counterfactual answers (the same question without the external
information), rewrites only the final `RELIED ON: ...` line to match the label, and fine-tunes with LoRA. No human
annotation is needed.

| Balanced accuracy of self-attribution (user suggestions) | Test set | Fresh set | Large set |
|---|---|---|---|
| Qwen3-8B, untrained → CREDIT | 0.49 → 0.86 | 0.51 → 0.86 | 0.53 → 0.79 |
| Qwen3-14B, untrained → CREDIT | 0.80 → 0.90 | 0.74 → 0.88 | 0.69 → 0.81 |
| DeepSeek-R1-Distill-Qwen-7B, untrained → CREDIT | 0.60 → 0.87 | 0.57 → 0.88 | 0.53 → 0.80 |
| DeepSeek-R1-Distill-Llama-8B, untrained → CREDIT | 0.62 → 0.85 | 0.62 → 0.79 | 0.59 → 0.79 |

## Repository layout

```
credit/                 Python package (data building, generation, judging, training, scoring)
scripts/                One script per pipeline stage (00-08) and reproduce_all.sh
data/annotations/       Human annotations: judge validation and failure analysis (annotators anonymized)
figures/                TikZ sources of the paper's figures
docs/PIPELINE.md        What each stage does, and which script produces each table and figure
docs/GLOSSARY.md        Internal names in the code and run tags, mapped to the terms used in the paper
data/README.md          The released data bundle: contents, sizes, layout, licenses
```

## Installation

```bash
pip install -e .            # CPU: data building, LLM judging, scoring
pip install -e ".[gpu]"     # GPU: generation (vLLM) and LoRA training (TRL/PEFT)
cp .env.example .env        # set CREDIT_ROOT and an OpenAI-compatible API endpoint for the judges
```

The LLM judges (attribution statement and answer matching), the generated passages and second wrong answers, and the
frontier-model evaluation call an OpenAI-compatible endpoint (`CREDIT_API_BASE`, `CREDIT_API_KEY`). The paper used
Gemini 3.8 Flash at temperature 0 as the judge. Generation uses vLLM with Qwen3's recommended sampling settings.

## Reproducing the paper

**From the released outputs (CPU, minutes).** Download the data from OSF and recompute the paper's numbers:

```bash
bash scripts/download_bundle.sh         # outputs, items, summaries from OSF (about 160 MB), checksums verified
bash scripts/08_paper_numbers.sh        # writes every result to $CREDIT_ROOT/numbers/
```

`docs/PIPELINE.md` lists which file holds which table or analysis.

**From scratch (GPU).** For one open model:

```bash
bash scripts/reproduce_all.sh qwen3-8b  # fetch data, build training data, train, evaluate, score
```

or stage by stage:

```bash
bash scripts/00_fetch_data.sh                      # public benchmarks
bash scripts/01_build_eval_sets.sh                 # optional: released item files are preferred
bash scripts/02_build_training_data.sh qwen3-8b    # generate, counterfactual answers, label, relabel
bash scripts/03_train.sh qwen3-8b                  # LoRA, rank 64, one epoch
bash scripts/04_evaluate.sh qwen3-8b base          # untrained model
bash scripts/04_evaluate.sh qwen3-8b cfdisc4L $CREDIT_ROOT/runs/qwen3-8b_cfdisc4L
bash scripts/05_competing.sh qwen3-8b cfdisc4L $CREDIT_ROOT/runs/qwen3-8b_cfdisc4L
bash scripts/06_frontier.sh gpt-5 claude-sonnet-5  # frontier models through the API (NOTES=1: system-prompt notes)
bash scripts/07_baselines.sh qwen3-8b              # VFT-style SFT, shuffled labels, faithfulness RL, prompting
bash scripts/08_paper_numbers.sh
```

Model keys: `qwen3-8b`, `qwen3-14b`, `r1-qwen-7b` (DeepSeek-R1-Distill-Qwen-7B), `r1-llama-8b`
(DeepSeek-R1-Distill-Llama-8B); see `credit/common.py`. Optional steps: `MULTI=1` for `02` builds the two-piece training
data, and `VARIANTS=1` for `04` adds template B, the neutral system-prompt note, the fresh-set transfer, and the
instruction-scope variant.
Generation and training ran on one 48-141 GB GPU per job. The pinned versions in `requirements.txt` are the ones
used for the paper; Transformers 5 or later is required (training reads `warmup_steps=0.05` as a 5% warmup ratio).

## Released data

Everything behind the paper's numbers is released: evaluation items, all model outputs with judge decisions and
influence labels, summary tables, and the human annotations. The annotations are in this repository; the larger files
are on OSF (`data/README.md`), together with the training sets and the LoRA adapters of the four CREDIT models.

## License

Code: Apache License 2.0 (`LICENSE`). The released data is derived from third-party datasets and model outputs that
remain under their own licenses and terms (`NOTICE`).

## Citation

```bibtex
@misc{zhou2026credit,
  title  = {Did the Model Use That? Counterfactual Training Boosts Self-Attribution in Language Models},
  author = {Zikai Zhou and Jiacheng Sang and Yilin Xu and Yufei Jin and Monica S. Lam},
  year   = {2026}
}
```
