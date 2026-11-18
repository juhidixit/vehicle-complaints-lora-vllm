"""QLoRA fine-tuning of the base model on the training split (PEFT + TRL SFTTrainer).

    python train/qlora.py --seed 0                      # GPU, settings from train/config.yaml
    python train/qlora.py --model <tiny> --no-4bit --max-steps 2 --limit 16   # CPU smoke test

Loss is computed on the answer (the label) only: the data is in TRL's conversational
prompt-completion format, for which SFTTrainer masks the prompt tokens.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from vclora.nhtsa import read_jsonl
from vclora.prompting import build_messages
from vclora.results import DATA, RESULTS, environment, load_labels


def to_example(row: dict, labels: list[str]) -> dict:
    return {
        "prompt": build_messages(row["text"], labels),
        "completion": [{"role": "assistant", "content": row["label"]}],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("train/config.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="override base_model (tests)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-4bit", action="store_true", help="plain LoRA, for CPU tests")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from vclora.generate import has_native_bf16

    cfg = yaml.safe_load(args.config.read_text())
    base = args.model or cfg["base_model"]
    out = args.out or Path(f"outputs/adapters/seed{args.seed}")
    labels = load_labels()
    train = read_jsonl(DATA / "train.jsonl")
    if args.limit:
        train = train[: args.limit]
    ds = Dataset.from_list([to_example(r, labels) for r in train])

    cuda = torch.cuda.is_available()
    bf16 = has_native_bf16()
    compute_dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    model_kwargs: dict = {"dtype": compute_dtype}
    use_4bit = cfg["quantization"]["load_in_4bit"] and not args.no_4bit
    if use_4bit:
        q = cfg["quantization"]
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=compute_dtype)
        model_kwargs["device_map"] = {"": 0}
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)

    t = cfg["training"]
    sft = SFTConfig(
        output_dir=str(out / "checkpoints"),
        seed=args.seed,
        num_train_epochs=t["num_train_epochs"],
        max_steps=args.max_steps,
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=t["warmup_ratio"],  # a float < 1 is a ratio of total steps in transformers 5
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        max_length=t["max_length"],
        optim=t["optim"] if use_4bit else "adamw_torch",
        gradient_checkpointing=t["gradient_checkpointing"] and cuda,
        logging_steps=t["logging_steps"],
        bf16=bf16,
        fp16=cuda and not bf16,
        completion_only_loss=True,
        packing=False,
        save_strategy="no",
        report_to=[],
        use_cpu=not cuda,
    )
    lora = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])
    trainer = SFTTrainer(model=model, args=sft, train_dataset=ds, processing_class=tok,
                         peft_config=lora)
    trainable, total = trainer.model.get_nb_trainable_parameters()

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = trainer.train()
    secs = time.perf_counter() - t0
    trainer.model.save_pretrained(out)
    tok.save_pretrained(out)

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    record = {
        "seed": args.seed,
        "base_model": base,
        "config": cfg,
        "qlora_4bit": use_4bit,
        "precision": "bf16" if bf16 else ("fp16" if cuda else "fp32"),
        "train_examples": len(ds),
        # TRL drops examples whose answer was cut off by max_length; report how many survived
        "train_examples_after_truncation": len(trainer.train_dataset),
        "steps": trainer.state.global_step,
        "train_seconds": secs,
        "train_loss": result.training_loss,
        "loss_curve": [[h["step"], h["loss"]] for h in trainer.state.log_history if "loss" in h],
        "final_logged_loss": losses[-1] if losses else None,
        "trainable_params": trainable,
        "total_params": total,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
        "adapter_dir": str(out),
        "environment": environment(),
    }
    (RESULTS / "training").mkdir(parents=True, exist_ok=True)
    (RESULTS / "training" / f"seed{args.seed}.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("config", "loss_curve")}, indent=1))


if __name__ == "__main__":
    main()
