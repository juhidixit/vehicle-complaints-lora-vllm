"""Upload the LoRA adapter (not the full model) and MODEL_CARD.md to the Hugging Face Hub.

    HF_TOKEN=... python train/push_adapter.py --adapter outputs/adapters/seed0 \
        --repo juhidixit/qwen2.5-1.5b-nhtsa-complaints-lora
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN (a write token) in the environment")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "adapter"
        shutil.copytree(args.adapter, dst, ignore=shutil.ignore_patterns("checkpoints"))
        shutil.copy(Path(__file__).resolve().parents[1] / "MODEL_CARD.md", dst / "README.md")
        api.upload_folder(folder_path=str(dst), repo_id=args.repo, repo_type="model",
                          commit_message="Upload QLoRA adapter and model card")
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
