"""Load test of the vLLM OpenAI-compatible server with `vllm bench serve`, on real test prompts.

    python bench/serve_load.py --config merged --model outputs/merged/seed0
    python bench/serve_load.py --config w4a16  --model outputs/w4a16/seed0
    python bench/serve_load.py --config lora   --model Qwen/Qwen2.5-1.5B-Instruct --adapter outputs/adapters/seed0

Starts `vllm serve`, replays the test split's rendered prompts at request rates 1, 4, 16 and
unlimited (Poisson arrivals), and writes one compact summary per configuration to
results/bench/serve_<config>.json. The raw `vllm bench serve` result files go to
results/bench/raw/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from vclora.generate import MAX_NEW_TOKENS, vllm_dtype
from vclora.nhtsa import read_jsonl
from vclora.prompting import build_messages
from vclora.results import DATA, RESULTS, environment, load_labels

RATES = ["1", "4", "16", "inf"]
NUM_PROMPTS = {"1": 200, "4": 400, "16": 1000, "inf": 1000}
KEEP = [
    "completed", "failed", "duration", "request_throughput", "output_throughput",
    "total_token_throughput", "mean_ttft_ms", "median_ttft_ms", "p95_ttft_ms", "mean_itl_ms",
    "median_itl_ms", "p95_itl_ms", "mean_tpot_ms", "median_tpot_ms", "p95_tpot_ms",
    "mean_e2el_ms", "median_e2el_ms", "p95_e2el_ms", "total_input_tokens", "total_output_tokens",
]


def wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            sys.exit(f"vllm serve exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(3)
    sys.exit("vllm serve did not become healthy in time")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--rates", nargs="+", default=RATES)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply prompt counts (smoke runs)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    labels = load_labels()
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = read_jsonl(DATA / "test.jsonl")
    raw_dir = RESULTS / "bench" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = Path("outputs") / "serve_prompts.jsonl"
    prompts_path.parent.mkdir(exist_ok=True)
    with open(prompts_path, "w") as fh:
        for r in rows:
            p = tok.apply_chat_template(build_messages(r["text"], labels), tokenize=False,
                                        add_generation_prompt=True)
            fh.write(json.dumps({"prompt": p}) + "\n")

    cmd = ["vllm", "serve", args.model, "--port", str(args.port), "--dtype", vllm_dtype(),
           "--max-model-len", "1024", "--gpu-memory-utilization", "0.85", "--seed", "0",
           "--served-model-name", "model"]
    if args.adapter:
        cmd += ["--enable-lora", "--max-lora-rank", "16", "--lora-modules", f"ft={args.adapter}"]
    log = open(raw_dir / f"server_{args.config}.log", "w")
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{args.port}"
    summary = {"config": args.config, "model": args.model, "adapter": args.adapter,
               "dtype": vllm_dtype(), "max_tokens": MAX_NEW_TOKENS, "server_cmd": cmd, "rates": {}}
    try:
        wait_healthy(base_url, server)
        for rate in args.rates:
            n = max(20, int(NUM_PROMPTS[rate] * args.scale))
            fname = f"serve_{args.config}_rate{rate}.json"
            bench = ["vllm", "bench", "serve", "--backend", "vllm", "--base-url", base_url,
                     "--model", "model", "--tokenizer", args.model, "--dataset-name", "custom",
                     "--dataset-path", str(prompts_path), "--skip-chat-template",
                     "--custom-output-len", str(MAX_NEW_TOKENS), "--num-prompts", str(n),
                     "--request-rate", rate, "--temperature", "0", "--seed", "0",
                     "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95",
                     "--save-result", "--result-dir", str(raw_dir), "--result-filename", fname,
                     "--disable-tqdm"]
            if args.adapter:
                bench += ["--lora-modules", "ft"]
            print(" ".join(bench), flush=True)
            subprocess.run(bench, check=True)
            res = json.loads((raw_dir / fname).read_text())
            summary["rates"][rate] = {k: res.get(k) for k in KEEP}
            summary["rates"][rate]["num_prompts"] = n
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
    summary["environment"] = environment()
    (RESULTS / "bench" / f"serve_{args.config}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["rates"], indent=1))


if __name__ == "__main__":
    main()
