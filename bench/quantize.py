"""4-bit weight-only quantization (W4A16, GPTQ) of the merged model with LLM Compressor.

    python bench/quantize.py --model outputs/merged/seed0 --out outputs/w4a16/seed0

The output is in compressed-tensors format, which vLLM loads directly. Calibration uses 256
training-split prompts with their answers, never test data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vclora.nhtsa import read_jsonl
from vclora.prompting import build_messages
from vclora.results import DATA, RESULTS, environment, load_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--scheme", default="W4A16")
    args = ap.parse_args()

    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    labels = load_labels()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")
    rows = read_jsonl(DATA / "train.jsonl")[: args.n_calib]
    texts = [
        tok.apply_chat_template(
            build_messages(r["text"], labels) + [{"role": "assistant", "content": r["label"]}],
            tokenize=False)
        for r in rows
    ]
    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tok(b["text"], add_special_tokens=False, truncation=True, max_length=512),
        batched=True, remove_columns=["text"])

    t0 = time.perf_counter()
    recipe = GPTQModifier(targets="Linear", scheme=args.scheme, ignore=["lm_head"])
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=512,
            num_calibration_samples=len(rows))
    secs = time.perf_counter() - t0
    model.save_pretrained(args.out, save_compressed=True)
    tok.save_pretrained(args.out)

    size = sum(f.stat().st_size for f in args.out.glob("*.safetensors"))
    src = sum(f.stat().st_size for f in args.model.glob("*.safetensors"))
    rec = {"model": str(args.model), "out": str(args.out), "scheme": args.scheme, "method": "GPTQ",
           "calibration_examples": len(rows), "seconds": secs, "weights_bytes": size,
           "source_weights_bytes": src, "environment": environment()}
    (RESULTS / "bench").mkdir(parents=True, exist_ok=True)
    (RESULTS / "bench" / f"quantize_{args.out.name}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "environment"}, indent=1))


if __name__ == "__main__":
    main()
