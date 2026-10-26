"""LoRA SFT on {prompt, completion} jsonl (loss on completion tokens only): CREDIT and the trained baselines."""
import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .common import MODELS


class FocusSFTTrainer(SFTTrainer):
    """Upweight the first `focus_tokens` completion tokens (the decision point right after a planted error) by
    `focus_weight`; the rest of the completion keeps weight 1. Loss = weighted mean over completion tokens.
    With focus_last, the LAST `focus_tokens` completion tokens are upweighted instead (e.g. the final RELIED ON line)."""

    def __init__(self, *a, focus_tokens=256, focus_weight=10.0, focus_last=False, **kw):
        super().__init__(*a, **kw)
        self.focus_tokens, self.focus_weight, self.focus_last = focus_tokens, focus_weight, focus_last
        self.model_accepts_loss_kwargs = False  # let Trainer divide by grad-accum steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        fwd = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask", "position_ids")}
        out = model(**fwd, use_cache=False)
        logits, tgt = out.logits[:, :-1], labels[:, 1:]
        valid = tgt != -100
        # rank of each valid target token within its completion (0 = first completion token)
        rank = valid.long().cumsum(-1) - 1
        if self.focus_last:  # count from the END of the completion instead (the final attribution line)
            rank = valid.long().sum(-1, keepdim=True) - 1 - rank
        w = torch.where(rank < self.focus_tokens, self.focus_weight, 1.0) * valid
        ce = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.clamp(min=0).reshape(-1),
                                               reduction="none").view_as(tgt)
        loss = (ce * w).sum() / w.sum().clamp(min=1)
        return (loss, out) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--bsz", type=int, default=16, help="effective batch size")
    ap.add_argument("--focus_tokens", type=int, default=0, help=">0: upweight the first N completion tokens")
    ap.add_argument("--focus_weight", type=float, default=10.0)
    ap.add_argument("--focus_last", action="store_true", help="with --focus_tokens N: upweight the LAST N completion tokens")
    ap.add_argument("--seed", type=int, default=42, help="training seed (LoRA init, data order); 42 = the original runs")
    ap.add_argument("--loss_from", default=None,
                    help="loss only on completion tokens from the LAST occurrence of this string on (e.g. 'RELIED ON:'); "
                         "examples without it are dropped")
    ap.add_argument("--mask_from", default=None,
                    help="the reverse of --loss_from: exclude completion tokens from the LAST occurrence of this string on "
                         "(e.g. 'RELIED ON:', for VFT with the attribution statement excluded from the loss)")
    args = ap.parse_args()
    assert not (args.loss_from and args.mask_from), "--loss_from and --mask_from are mutually exclusive"
    cfg = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_dataset("json", data_files=args.data, split="train").shuffle(seed=0 if args.seed == 42 else args.seed)
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "completion")])
    # Tokenize prompt and completion SEPARATELY (as at inference: the prefix is tokenized alone, then the model generates).
    # Joint tokenization merges a trailing prompt space with the first completion word (" Wait"), so that decisive token
    # gets masked as prompt and is never trained.
    def tok_fn(e):
        p = tok(e["prompt"], add_special_tokens=False)["input_ids"]
        if args.mask_from:
            k = e["completion"].rfind(args.mask_from)
            if k >= 0:
                c0 = tok(e["completion"][:k], add_special_tokens=False)["input_ids"]
                c1 = tok(e["completion"][k:], add_special_tokens=False)["input_ids"]
                return {"input_ids": p + c0 + c1, "completion_mask": [0] * len(p) + [1] * len(c0) + [0] * len(c1)}
        if args.loss_from:
            k = e["completion"].rfind(args.loss_from)
            if k < 0:
                return {"input_ids": p, "completion_mask": [0] * len(p)}  # no marker: dropped by the filter below
            # the marker follows a newline, so tokenizing the two parts separately does not change the token boundary
            c0 = tok(e["completion"][:k], add_special_tokens=False)["input_ids"]
            c1 = tok(e["completion"][k:], add_special_tokens=False)["input_ids"]
            return {"input_ids": p + c0 + c1, "completion_mask": [0] * (len(p) + len(c0)) + [1] * len(c1)}
        c = tok(e["completion"], add_special_tokens=False)["input_ids"]
        return {"input_ids": p + c, "completion_mask": [0] * len(p) + [1] * len(c)}
    ds = ds.map(tok_fn, remove_columns=["prompt", "completion"])
    n0 = len(ds)
    # DROP over-length examples instead of truncating them: truncation cuts the end of the completion (final answer, EOS)
    ds = ds.filter(lambda e: len(e["input_ids"]) <= args.max_len and sum(e["completion_mask"]) > 0)
    print(f"[sft] {len(ds)}/{n0} examples kept; {n0 - len(ds)} dropped as longer than {args.max_len} tokens", flush=True)
    e0 = ds[0]
    first = e0["completion_mask"].index(1)
    print("[sft] first trained tokens:", tok.convert_ids_to_tokens(e0["input_ids"][first:first + 4]), flush=True)
    if args.loss_from:
        n_tr = [sum(e["completion_mask"]) for e in ds.select(range(min(200, len(ds))))]
        print(f"[sft] loss_from={args.loss_from!r}: trained tokens per example (first 200) mean {sum(n_tr)/len(n_tr):.1f} "
              f"max {max(n_tr)} | example: {tok.decode(e0['input_ids'][first:])!r}", flush=True)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    per_dev = 1
    conf = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=0.05, per_device_train_batch_size=per_dev, gradient_accumulation_steps=max(1, args.bsz // (per_dev * world)),
        bf16=True, gradient_checkpointing=True, logging_steps=5, save_strategy="no", report_to=[],
        max_length=args.max_len, completion_only_loss=True, dataset_kwargs={"add_special_tokens": False},
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"}, seed=args.seed, data_seed=args.seed,
    )
    peft = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    kw = dict(model=cfg["path"], args=conf, train_dataset=ds, processing_class=tok, peft_config=peft)
    tr = (FocusSFTTrainer(**kw, focus_tokens=args.focus_tokens, focus_weight=args.focus_weight, focus_last=args.focus_last) if args.focus_tokens
          else SFTTrainer(**kw))
    tr.train()
    tr.save_model(args.out)
    tok.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
