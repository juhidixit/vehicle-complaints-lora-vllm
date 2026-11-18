"""Merge a LoRA adapter into the full-precision base model, for serving without LoRA overhead.

    python train/merge.py --adapter outputs/adapters/seed0 --out outputs/merged/seed0

The adapter was trained on a 4-bit base (QLoRA) but is merged into the 16-bit base, which is the
usual practice; the small train/serve mismatch this creates is measured, not assumed, because
the merged model is what eval/llm_eval.py scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vclora.generate import resolve_dtype

    base = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(base, dtype=resolve_dtype(args.dtype))
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"merged {args.adapter} into {base} -> {args.out}")


if __name__ == "__main__":
    main()
