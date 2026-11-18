"""End-to-end on CPU: classic baselines, prompting baselines, LoRA training, merge, scoring, report."""

import json
import shutil
import subprocess
import sys

import pytest

from .conftest import ROOT

pytest.importorskip("trl")


def run(args, workspace, cwd=None):
    subprocess.run([sys.executable, *args], check=True, env=workspace["env"], cwd=cwd or ROOT)


def test_classic_baselines(workspace):
    run([str(ROOT / "eval/baselines_classic.py")], workspace)
    res = workspace["ws"] / "results/metrics"
    tf = json.loads((res / "tfidf_logreg__test.json").read_text())
    maj = json.loads((res / "majority__test.json").read_text())
    assert tf["n"] == 100 and len(tf["val_search"]) == 6
    assert tf["macro_f1"] > maj["macro_f1"]  # synthetic classes are separable by keywords


def test_llm_systems_on_tiny_model(workspace, tiny_model, tmp_path):
    ws = workspace["ws"]
    run([str(ROOT / "eval/llm_eval.py"), "--system", "zero_shot", "--model", str(tiny_model),
         "--engine", "hf", "--limit", "8", "--batch-size", "4"], workspace)
    run([str(ROOT / "eval/llm_eval.py"), "--system", "five_shot", "--model", str(tiny_model),
         "--engine", "hf", "--limit", "8", "--shots", "5"], workspace)
    adapter = tmp_path / "adapter"
    run([str(ROOT / "train/qlora.py"), "--model", str(tiny_model), "--no-4bit", "--max-steps", "2",
         "--limit", "16", "--out", str(adapter)], workspace)
    assert (adapter / "adapter_config.json").exists()
    rec = json.loads((ws / "results/training/seed0.json").read_text())
    assert rec["steps"] == 2 and rec["trainable_params"] > 0 and not rec["qlora_4bit"]
    assert rec["train_examples_after_truncation"] == 16

    merged = tmp_path / "merged"
    run([str(ROOT / "train/merge.py"), "--adapter", str(adapter), "--out", str(merged)], workspace)
    run([str(ROOT / "eval/llm_eval.py"), "--system", "qlora_seed0", "--model", str(merged),
         "--engine", "hf", "--limit", "8"], workspace)
    # the adapter served unmerged must give the same predictions as the merged model
    run([str(ROOT / "eval/llm_eval.py"), "--system", "qlora_seed0_unmerged", "--model", str(tiny_model),
         "--adapter", str(adapter), "--engine", "hf", "--limit", "8"], workspace)
    preds = {}
    for s in ("qlora_seed0", "qlora_seed0_unmerged"):
        lines = (ws / f"results/predictions/{s}__test.jsonl").read_text().splitlines()
        preds[s] = [json.loads(x)["pred"] for x in lines]
    assert preds["qlora_seed0"] == preds["qlora_seed0_unmerged"]
    m = json.loads((ws / "results/metrics/qlora_seed0__test.json").read_text())
    assert m["n"] == 8 and 0 <= m["invalid_rate"] <= 1 and m["limit"] == 8


def test_report_renders_all_sections(workspace, tmp_path):
    # copy the repo's report script and README into a scratch root that has these results
    root = tmp_path / "repo"
    (root / "report").mkdir(parents=True)
    (root / "bench").mkdir()
    shutil.copy(ROOT / "report/make_report.py", root / "report")
    shutil.copy(ROOT / "README.md", root)
    shutil.copy(ROOT / "bench/prices.yaml", root / "bench")
    shutil.copytree(workspace["ws"] / "results", root / "results")
    run([str(root / "report/make_report.py")], workspace, cwd=root)
    text = (root / "README.md").read_text()
    assert "TF-IDF + logistic regression" in text
    assert "Majority class" in text
    run([str(root / "report/make_report.py"), "--check"], workspace, cwd=root)


def test_loss_is_on_the_answer_only(tiny_model):
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from vclora.prompting import build_messages

    tok = AutoTokenizer.from_pretrained(tiny_model)
    ex = {"prompt": build_messages("brakes pedal soft grinding " * 6, ["SERVICE BRAKES", "ENGINE"]),
          "completion": [{"role": "assistant", "content": "SERVICE BRAKES"}]}
    trainer = SFTTrainer(
        model=AutoModelForCausalLM.from_pretrained(tiny_model),
        args=SFTConfig(output_dir="/tmp/vcl-sft-test", completion_only_loss=True, report_to=[],
                       use_cpu=True, max_steps=1),
        train_dataset=Dataset.from_list([ex]), processing_class=tok,
        peft_config=LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear"))
    batch = trainer.data_collator([trainer.train_dataset[0]])
    ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
    trained = tok.decode([i for i, lab in zip(ids, labels, strict=True) if lab != -100])
    assert trained.strip() == "SERVICE BRAKES<|im_end|>"
