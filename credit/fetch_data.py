"""Download the public benchmarks used in the paper into $CREDIT_ROOT/data/bench (file names as the builders expect).

  sycophancy-eval  answer subset (Sharma et al.)       git clone  -> bench/sycophancy-eval/datasets/answer.jsonl
  FaithEval        counterfactual v1.0 (Salesforce)     HF         -> bench/Salesforce__FaithEval-counterfactual-v1.0__test.jsonl
  RAMDocs          (Wang et al.)                        HF         -> bench/HanNight__RAMDocs__test.jsonl
  PCBench          (only used by the pub_eval builder)  HF         -> bench/ALIENS232__PCBench__train.jsonl
TriviaQA, MMLU, ARC, and CommonsenseQA are loaded on demand through `datasets` by the training-data builders.
Usage: CREDIT_ROOT=./workdir python -m credit.fetch_data
"""
import os
import subprocess


def main():
    out = os.path.join(os.environ.get("CREDIT_ROOT", "."), "data", "bench")
    os.makedirs(out, exist_ok=True)
    from datasets import load_dataset
    for name, split in (("Salesforce/FaithEval-counterfactual-v1.0", "test"), ("HanNight/RAMDocs", "test"),
                        ("ALIENS232/PCBench", "train")):
        path = os.path.join(out, f"{name.replace('/', '__')}__{split}.jsonl")
        if not os.path.exists(path):
            load_dataset(name, split=split).to_json(path)
        print("ok", path)
    syco = os.path.join(out, "sycophancy-eval")
    if not os.path.exists(syco):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/meg-tong/sycophancy-eval.git", syco], check=True)
    print("ok", os.path.join(syco, "datasets", "answer.jsonl"))


if __name__ == "__main__":
    main()
