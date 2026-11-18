"""Score an LLM system (prompted base model, fine-tuned or quantized) on the test splits.

    python eval/llm_eval.py --system zero_shot  --model Qwen/Qwen2.5-1.5B-Instruct
    python eval/llm_eval.py --system five_shot  --model Qwen/Qwen2.5-1.5B-Instruct --shots 5
    python eval/llm_eval.py --system qlora_seed0 --model outputs/merged/seed0
    python eval/llm_eval.py --system qlora_seed0_w4a16 --model outputs/w4a16/seed0

Output is parsed against the closed label list; unparseable output counts as wrong and its rate
is reported as ``invalid_rate``.
"""

from __future__ import annotations

import argparse

from vclora.generate import HFGenerator, VLLMGenerator, render, vllm_dtype
from vclora.nhtsa import read_jsonl
from vclora.prompting import build_messages, parse_label, pick_shots
from vclora.results import DATA, SPLITS, environment, load_labels, save_run

SHOT_SEED = 7


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (served as a LoRA, not merged)")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows (smoke test)")
    ap.add_argument("--batch-size", type=int, default=16, help="HF engine only")
    ap.add_argument("--family", default="llm")
    args = ap.parse_args()

    labels = load_labels()
    shots = pick_shots(read_jsonl(DATA / "train.jsonl"), args.shots, SHOT_SEED) if args.shots else []
    if args.engine == "vllm":
        gen = VLLMGenerator(args.model, adapter=args.adapter, dtype=vllm_dtype())
    else:
        gen = HFGenerator(args.model, adapter=args.adapter, batch_size=args.batch_size)

    for split in args.splits:
        rows = read_jsonl(DATA / f"{split}.jsonl")
        if args.limit:
            rows = rows[: args.limit]
        prompts = render(gen.tok, [build_messages(r["text"], labels, shots) for r in rows])
        res = gen.generate(prompts)
        preds = [parse_label(t, labels) for t in res.texts]
        save_run(args.system, split, rows, preds, labels, raw=res.texts, extra={
            "family": args.family,
            "model": args.model,
            "adapter": args.adapter,
            "shots": args.shots,
            "shot_odinos": [s["odino"] for s in shots],
            "engine": args.engine,
            "limit": args.limit,
            "generation_seconds": res.seconds,
            "classifications_per_second": len(rows) / res.seconds,
            "mean_prompt_tokens": res.prompt_tokens / len(rows),
            "mean_output_tokens": res.output_tokens / len(rows),
            "environment": environment(),
        })


if __name__ == "__main__":
    main()
