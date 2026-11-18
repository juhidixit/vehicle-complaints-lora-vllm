---
library_name: peft
base_model: Qwen/Qwen2.5-1.5B-Instruct
license: apache-2.0
language: [en]
tags: [lora, qlora, text-classification, nhtsa, automotive]
pipeline_tag: text-generation
---

# Qwen2.5-1.5B-Instruct QLoRA adapter: NHTSA complaint component classifier

A LoRA adapter that makes `Qwen/Qwen2.5-1.5B-Instruct` answer with the failed vehicle component
for a US vehicle-safety complaint narrative. Code, data card and every measured result:
https://github.com/juhidixit/vehicle-complaints-lora-vllm

## Intended use

Research and portfolio demonstration of fine-tuning vs prompting on a closed-label task. Not for
safety decisions: it routes text, it does not judge whether a defect exists.

## Training data

NHTSA Office of Defects Investigation consumer complaints (public domain), single-component
complaints only, 10 most frequent top-level components, 20,000 training examples. Split IDs are
in the GitHub repo under `data/splits/`.

## Training procedure

QLoRA: 4-bit NF4 base with double quantization, LoRA r=16, alpha=32, dropout 0.05 on all linear
layers, learning rate 2e-4 with cosine decay, 2 epochs, effective batch 16, max length 512, loss on
the answer tokens only. Exact settings: `train/config.yaml` in the repo. Hardware, run time and
seed: `results/training/` in the repo.

## How to use

Use the same system prompt as training (`src/vclora/prompting.py` in the repo): it lists the 10
labels and asks for exactly one. With vLLM: `vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-lora
--lora-modules nhtsa=<this repo>`.

## Evaluation

Macro-F1, accuracy and per-class F1 with 95% bootstrap confidence intervals, on a held-out test set
and on a later-year time-shift test set, against TF-IDF + logistic regression and zero-/5-shot
prompting of the same base model. See the Results section of the GitHub README; numbers are not
copied here so they cannot drift.

## Limitations

Labels are chosen by the people complaining and are noisy. Only single-component complaints were
used. English only. The model can still output text outside the label list; callers must map
output to the closed list and treat anything else as no answer.
