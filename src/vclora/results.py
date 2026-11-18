"""Writing predictions, metrics and environment info under results/."""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
import os
import platform
import subprocess
from pathlib import Path

from vclora.metrics import score

RESULTS = Path(os.environ.get("VCL_RESULTS", "results"))
DATA = Path(os.environ.get("VCL_DATA", "data/processed"))
SPLITS = ("test", "test_timeshift")


def load_labels() -> list[str]:
    return json.loads((DATA / "labels.json").read_text())


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def environment() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": {
            p: _version(p)
            for p in ("torch", "transformers", "peft", "trl", "bitsandbytes", "vllm", "llmcompressor",
                      "scikit-learn", "accelerate")
            if _version(p)
        },
        "date_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()
        env["gpus"] = [line.strip() for line in out]
    except (OSError, subprocess.SubprocessError):
        env["gpus"] = []
    try:
        import torch

        env["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return env


def save_run(system: str, split: str, rows: list[dict], preds: list[str], labels: list[str],
             raw: list[str] | None = None, extra: dict | None = None) -> dict:
    """rows: the split's examples; preds: parsed predictions in the same order."""
    pred_dir, met_dir = RESULTS / "predictions", RESULTS / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / f"{system}__{split}.jsonl", "w", encoding="utf-8") as fh:
        for i, (r, p) in enumerate(zip(rows, preds, strict=True)):
            rec = {"odino": r["odino"], "label": r["label"], "pred": p}
            if raw is not None:
                rec["raw"] = raw[i][:80]
            fh.write(json.dumps(rec) + "\n")
    m = score([r["label"] for r in rows], preds, labels)
    m.update({"system": system, "split": split, **(extra or {})})
    (met_dir / f"{system}__{split}.json").write_text(json.dumps(m, indent=1) + "\n")
    print(f"{system:28s} {split:15s} macro-F1 {m['macro_f1']:.3f} "
          f"[{m['macro_f1_ci95'][0]:.3f}, {m['macro_f1_ci95'][1]:.3f}]  acc {m['accuracy']:.3f}  "
          f"invalid {m['invalid_rate']:.1%}")
    return m
