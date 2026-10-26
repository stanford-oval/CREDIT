"""Shared utilities: datasets, prompting, generation, answer extraction."""
import json
import os
import random
import re
from dataclasses import dataclass, field, asdict

LETTERS = "ABCDEFGHIJ"

# Models under study. `reasoning` models emit <think>...</think> before the answer.
MODELS = {
    "qwen3-8b": dict(path="Qwen/Qwen3-8B", reasoning=True),
    "qwen3-8b-base": dict(path="Qwen/Qwen3-8B-Base", reasoning=True),  # pretrained-only counterpart
    "r1-qwen-7b": dict(path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", reasoning=True),
    "r1-qwen-14b": dict(path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", reasoning=True),
    "r1-llama-8b": dict(path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", reasoning=True),
    "qwen25-7b": dict(path="Qwen/Qwen2.5-7B-Instruct", reasoning=False),
    "qwen25-1.5b": dict(path="Qwen/Qwen2.5-1.5B-Instruct", reasoning=False),
    "qwen3-1.7b-nothink": dict(path="Qwen/Qwen3-1.7B", reasoning=False, qwen3_nothink=True),
    "qwen3-8b-nothink": dict(path="Qwen/Qwen3-8B", reasoning=False, qwen3_nothink=True),
    "qwen25-32b": dict(path="Qwen/Qwen2.5-32B-Instruct", reasoning=False),
    "qwen3-14b": dict(path="Qwen/Qwen3-14B", reasoning=True),
    "qwen3-1.7b": dict(path="Qwen/Qwen3-1.7B", reasoning=True),
    "qwen3-32b": dict(path="Qwen/Qwen3-32B", reasoning=True),
    "phi35-mini": dict(path="microsoft/Phi-3.5-mini-instruct", reasoning=False),
}


@dataclass
class Item:
    id: str
    dataset: str
    question: str
    options: list
    answer: str  # gold letter
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _mk(ds, i, q, opts, ans, **meta):
    return Item(id=f"{ds}-{i}", dataset=ds, question=q.strip(), options=[o.strip() for o in opts], answer=ans, meta=meta)


def load_dataset_items(name, n=None, seed=0, split=None):
    """Load a multiple-choice dataset as a list of Item."""
    from datasets import load_dataset

    items = []
    if name == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split or "test")
        for i, r in enumerate(ds):
            labels = r["choices"]["label"]
            if r["answerKey"] not in labels:
                continue
            ans = LETTERS[labels.index(r["answerKey"])]
            items.append(_mk(name, i, r["question"], r["choices"]["text"], ans))
    elif name == "csqa":
        ds = load_dataset("tau/commonsense_qa", split=split or "validation")
        for i, r in enumerate(ds):
            ans = LETTERS[r["choices"]["label"].index(r["answerKey"])]
            items.append(_mk(name, i, r["question"], r["choices"]["text"], ans))
    elif name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split=split or "test")
        for i, r in enumerate(ds):
            items.append(_mk(name, i, r["question"], r["choices"], LETTERS[r["answer"]], subject=r["subject"]))
    elif name == "aqua":
        ds = load_dataset("deepmind/aqua_rat", "raw", split=split or "test")
        for i, r in enumerate(ds):
            opts = [re.sub(r"^[A-E]\)", "", o) for o in r["options"]]
            items.append(_mk(name, i, r["question"], opts, r["correct"]))
    elif name == "logiqa":
        ds = load_dataset("lucasmccabe/logiqa", split=split or "test", revision="refs/convert/parquet")
        for i, r in enumerate(ds):
            items.append(_mk(name, i, r["context"] + "\n" + r["query"], r["options"], LETTERS[r["correct_option"]]))
    elif name == "gpqa":
        # Gated on HF; falls back gracefully if unavailable.
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        rng = random.Random(seed)
        for i, r in enumerate(ds):
            opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            perm = list(range(4))
            rng.shuffle(perm)
            opts = [opts[p] for p in perm]
            items.append(_mk(name, i, r["Question"], opts, LETTERS[perm.index(0)]))
    else:
        raise ValueError(name)
    if n is not None and n < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items


def format_question(item, with_options=True):
    s = item.question
    if with_options:
        s += "\n\nAnswer choices:\n" + "\n".join(f"({LETTERS[j]}) {o}" for j, o in enumerate(item.options))
    return s


COT_INSTRUCTION = (
    "Think step by step, then give your final answer on the last line in the form "
    "\"Answer: (X)\" where X is the letter of the correct choice."
)


def user_prompt(item, prefix="", suffix=""):
    body = format_question(item)
    parts = [p for p in [prefix, body, suffix, COT_INSTRUCTION] if p]
    return "\n\n".join(parts)


def chat_prefix(tokenizer, user_msg, model_cfg, system=None):
    """Render the chat template up to the start of the assistant turn (and open <think> for reasoning models)."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_msg}]
    kw = {}
    if model_cfg.get("qwen3_nothink"):
        kw["enable_thinking"] = False
    elif "Qwen3" in model_cfg["path"]:
        kw["enable_thinking"] = True
    s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    if model_cfg["reasoning"] and not s.rstrip().endswith("<think>"):
        s += "<think>\n"
    return s


# The letter must stand alone (\b after it). Without it, "### Final Answer:\nAnswer: (D)" matched "Answer:\nA" (the A of the
# next word "Answer"), which also swallowed the real "Answer: (D)" -> recorded answer A.
_ANS_RE = [
    re.compile(r"Answer\s*[:：]\s*\**\s*\(?([A-J])\b\)?", re.I),
    re.compile(r"answer is\s*:?\s*\**\s*\(?([A-J])\)", re.I),
    re.compile(r"\\boxed\{\s*\(?([A-J])\)?\s*\}"),
]


def extract_answer(text, n_options=10):
    """Return last answer letter found in text, or None."""
    valid = LETTERS[:n_options]
    for rx in _ANS_RE:
        ms = [m.group(1).upper() for m in rx.finditer(text)]
        ms = [m for m in ms if m in valid]
        if ms:
            return ms[-1]
    return None


def split_reasoning(text, reasoning):
    """Split a completion into (cot, post) where post is the visible answer section."""
    if reasoning:
        if "</think>" in text:
            cot, post = text.split("</think>", 1)
            return cot.strip(), post.strip()
        return text.strip(), ""
    # non-reasoning: CoT is everything before the final "Answer:" line
    m = list(re.finditer(r"Answer\s*[:：]", text, re.I))
    if m:
        return text[: m[-1].start()].strip(), text[m[-1].start():].strip()
    return text.strip(), ""


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\\$*-])|\n+")


def split_sentences(text):
    """Split CoT into steps (sentences / lines), keeping non-empty chunks."""
    parts = [p.strip() for p in _SENT_RE.split(text)]
    return [p for p in parts if p]


def load_llm(model_cfg, max_model_len=16384, tp=None, gpu_mem=0.9, **kw):
    from vllm import LLM

    tp = tp or int(os.environ.get("TP", "1"))
    return LLM(
        model=model_cfg["path"],
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed=0,
        **kw,
    )


# Evaluation decoding for Qwen3 thinking mode. Greedy decoding causes endless repetition loops in thinking mode (Qwen3 docs);
# on our evals ~95% of truncated responses were such loops. Use the recommended sampling with a fixed seed instead.
EVAL_DECODING = dict(temperature=0.6, top_p=0.95, top_k=20, seed=1234)


def sampling(max_tokens=8192, temperature=0.6, top_p=0.95, n=1, seed=None, **kw):
    from vllm import SamplingParams

    return SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed, **kw)


def answer_logprob_prompt(model_cfg, prefix_text, cot):
    """Prompt that forces an immediate answer after (possibly truncated) CoT."""
    if model_cfg["reasoning"]:
        return prefix_text + cot + "\n</think>\n\nAnswer: ("
    return prefix_text + cot + "\n\nAnswer: ("


def letter_distribution(llm, prompts, n_options):
    """Probability over option letters for the next token after each prompt."""
    import math
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    res = []
    for o in outs:
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        probs = {L: 0.0 for L in LETTERS[:n_options]}
        for tok_id, info in lp.items():
            t = (info.decoded_token or "").strip().lstrip("(").upper()
            if t in probs:
                probs[t] += math.exp(info.logprob)
        z = sum(probs.values())
        if z > 0:
            probs = {k: v / z for k, v in probs.items()}
        res.append(probs)
    return res


def argmax_letter(probs):
    if not probs or max(probs.values()) == 0:
        return None
    return max(probs, key=probs.get)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def results_dir():
    return os.environ.get("CREDIT_RESULTS", os.path.join(os.environ.get("CREDIT_ROOT", "."), "results"))
