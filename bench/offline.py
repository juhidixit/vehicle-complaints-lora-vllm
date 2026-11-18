"""Offline throughput: HF Transformers generate vs vLLM, on real test-split prompts.

    python bench/offline.py --config hf_bs8      --model outputs/merged/seed0
    python bench/offline.py --config vllm        --model outputs/merged/seed0
    python bench/offline.py --config vllm_w4a16  --model outputs/w4a16/seed0
    python bench/offline.py --config vllm_lora   --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/adapters/seed0

Run one configuration per process so GPU memory from one engine never leaks into the next.
Peak GPU memory is sampled from NVML (whole device), so for vLLM it includes the KV-cache pool
that vLLM pre-allocates (gpu_memory_utilization), not just the weights.
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np

from vclora.generate import HFGenerator, VLLMGenerator, render, vllm_dtype
from vclora.nhtsa import read_jsonl
from vclora.prompting import build_messages
from vclora.results import DATA, RESULTS, environment, load_labels


class GpuMemSampler:
    """Peak device memory (GiB) via NVML, sampled every 50 ms in a background thread."""

    def __init__(self):
        self.peak = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml = pynvml
        except Exception:  # noqa: BLE001 - no GPU / no NVML: report None
            self._nvml = None

    def __enter__(self):
        if self._nvml:
            self.peak = 0.0
            self._stop = threading.Event()
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            used = self._nvml.nvmlDeviceGetMemoryInfo(self._h).used / 2**30
            self.peak = max(self.peak, used)
            time.sleep(0.05)

    def __exit__(self, *exc):
        if self._nvml:
            self._stop.set()
            self._t.join()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="hf_bs1 | hf_bs8 | hf_bs32 | vllm | vllm_w4a16 | vllm_lora")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=0, help="prompts to run (default: whole test split)")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    labels = load_labels()
    rows = read_jsonl(DATA / f"{args.split}.jsonl")
    is_hf = args.config.startswith("hf_")
    batch = int(args.config.split("_bs")[1]) if is_hf else None
    n = args.n or (256 if batch == 1 else len(rows))
    rows = rows[:n]

    with GpuMemSampler() as mem:
        t_load = time.perf_counter()
        if is_hf:
            gen = HFGenerator(args.model, adapter=args.adapter, batch_size=batch)
        else:
            gen = VLLMGenerator(args.model, adapter=args.adapter, dtype=vllm_dtype())
        load_s = time.perf_counter() - t_load
        prompts = render(gen.tok, [build_messages(r["text"], labels) for r in rows])
        gen.generate(prompts[: max(8, batch or 8)])  # warm-up, not timed
        res = gen.generate(prompts)
        batch_lat = None
        if is_hf:  # per-batch latency on a slice, for the batch-1 latency story
            lats = []
            for i in range(0, min(len(prompts), 32 * batch), batch):
                t0 = time.perf_counter()
                gen.generate(prompts[i : i + batch])
                lats.append(time.perf_counter() - t0)
            batch_lat = {"p50_ms": float(np.percentile(lats, 50) * 1000),
                         "p95_ms": float(np.percentile(lats, 95) * 1000), "batches": len(lats)}

    torch_peak = None
    if is_hf:
        import torch

        if torch.cuda.is_available():
            torch_peak = torch.cuda.max_memory_allocated() / 2**30

    rec = {
        "config": args.config,
        "engine": "hf" if is_hf else "vllm",
        "batch_size": batch,
        "model": args.model,
        "adapter": args.adapter,
        "dtype": vllm_dtype(),
        "split": args.split,
        "requests": len(rows),
        "load_seconds": load_s,
        "seconds": res.seconds,
        "requests_per_second": len(rows) / res.seconds,
        "output_tokens_per_second": res.output_tokens / res.seconds,
        "total_tokens_per_second": (res.output_tokens + res.prompt_tokens) / res.seconds,
        "mean_prompt_tokens": res.prompt_tokens / len(rows),
        "mean_output_tokens": res.output_tokens / len(rows),
        "batch_latency": batch_lat,
        "peak_gpu_memory_gb_nvml": mem.peak,
        "peak_gpu_memory_gb_torch": torch_peak,
        "environment": environment(),
    }
    (RESULTS / "bench").mkdir(parents=True, exist_ok=True)
    (RESULTS / "bench" / f"offline_{args.config}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "environment"}, indent=1))


if __name__ == "__main__":
    main()
