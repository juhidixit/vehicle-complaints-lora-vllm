"""Generate the README result tables and the charts from results/*.json.

    python report/make_report.py           # rewrite README sections + results/figures/*.png
    python report/make_report.py --check   # fail if README is out of date (used in CI)

Every number in the README between <!-- BEGIN:name --> and <!-- END:name --> markers comes from
here. Nothing in those sections is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
README = ROOT / "README.md"
NOT_RUN = "_Not run yet._"

SYSTEM_NAMES = {
    "majority": "Majority class",
    "tfidf_logreg": "TF-IDF + logistic regression",
    "zero_shot": "Qwen2.5-1.5B-Instruct, zero-shot",
    "five_shot": "Qwen2.5-1.5B-Instruct, 5-shot",
}
ORDER = ["majority", "tfidf_logreg", "zero_shot", "five_shot"]
SERVE_NAMES = {"merged": "vLLM, merged 16-bit", "w4a16": "vLLM, merged W4A16 (GPTQ)",
               "lora": "vLLM, base + LoRA adapter"}
OFFLINE_ORDER = ["hf_bs1", "hf_bs8", "hf_bs32", "vllm", "vllm_lora", "vllm_w4a16"]
OFFLINE_NAMES = {"hf_bs1": "HF Transformers, batch 1", "hf_bs8": "HF Transformers, batch 8",
                 "hf_bs32": "HF Transformers, batch 32", "vllm": "vLLM, merged 16-bit",
                 "vllm_lora": "vLLM, base + LoRA adapter", "vllm_w4a16": "vLLM, merged W4A16"}
# which scored system's quality goes with which serving configuration
CONFIG_SYSTEM = {"vllm": "qlora_seed0", "vllm_lora": "qlora_seed0", "hf_bs32": "qlora_seed0",
                 "hf_bs8": "qlora_seed0", "hf_bs1": "qlora_seed0", "vllm_w4a16": "qlora_seed0_w4a16"}


def load(pattern: str) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(RES.glob(pattern))}


def name(system: str) -> str:
    if system in SYSTEM_NAMES:
        return SYSTEM_NAMES[system]
    m = re.fullmatch(r"qlora_seed(\d+)(_w4a16)?", system)
    if m:
        return f"QLoRA fine-tuned, seed {m.group(1)}" + (", W4A16 (GPTQ)" if m.group(2) else "")
    return system


def f1cell(m: dict | None) -> str:
    if not m:
        return "–"
    lo, hi = m["macro_f1_ci95"]
    return f"{m['macro_f1']:.3f} ({lo:.3f}–{hi:.3f})"


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def sort_key(system: str) -> tuple:
    return (ORDER.index(system) if system in ORDER else len(ORDER), system)


# ---------------------------------------------------------------- sections

def section_data() -> str:
    p = RES / "data_stats.json"
    if not p.exists():
        return NOT_RUN
    d = json.loads(p.read_text())
    src = d.get("source", {})
    f = d["funnel"]
    lines = [
        f"Source: `{src.get('url', 'n/a')}`, downloaded {src.get('downloaded_at_utc', 'n/a')} "
        f"(server Last-Modified: {src.get('last_modified', 'n/a')}, SHA-256 `{str(src.get('sha256', ''))[:16]}…`). "
        f"Latest complaint received: {d['latest_received_date']}.",
        "",
        f"- Main pool: complaints received {d['pool_years'][0]}–{d['pool_years'][1]}. "
        f"Time-shift test: complaints received in {d['timeshift_year']}.",
        f"- Flat-file lines read: {f.get('lines', 0):,}; complaints in the date window: "
        f"{f.get('complaints_in_date_range', 0):,}.",
        f"- Pool after filters: {f.get('pool_kept', 0):,} complaints (dropped: "
        f"{f.get('pool_dropped_multi_or_unknown_component', 0):,} with several top-level components or "
        f"only UNKNOWN OR OTHER, {f.get('pool_dropped_short', 0):,} under {d['min_words']} words, "
        f"{f.get('pool_dropped_duplicate_text', 0):,} exact-text duplicates, "
        f"{f.get('pool_dropped_duplicate_text_conflicting_label', 0):,} duplicates with conflicting labels). "
        f"{f.get('pool_top10_available', 0):,} of them are in the 10 most frequent components.",
        f"- Time-shift year after filters: {f.get('timeshift_kept', 0):,} complaints, "
        f"{f.get('timeshift_top10_available', 0):,} in the 10 labels.",
        "",
    ]
    splits = d["splits"]
    names = list(splits)
    rows = []
    for lab in d["labels"]:
        rows.append([lab] + [f"{splits[s]['labels'].get(lab, 0):,}" for s in names])
    rows.append(["**Total**"] + [f"**{splits[s]['n']:,}**" for s in names])
    rows.append(["Median / p95 words"] + [f"{splits[s]['words_median']:.0f} / {splits[s]['words_p95']}" for s in names])
    lines.append(table(["Component (label)", *names], rows))
    return "\n".join(lines)


def section_results(metrics: dict[str, dict]) -> str:
    systems = sorted({k.split("__")[0] for k in metrics}, key=sort_key)
    if not systems:
        return NOT_RUN
    rows = []
    for s in systems:
        t, ts = metrics.get(f"{s}__test"), metrics.get(f"{s}__test_timeshift")
        lim = (t or ts or {}).get("limit")
        label = name(s) + (f" (first {lim} rows only)" if lim else "")
        rows.append([
            label, f1cell(t), pct(t and t["accuracy"]), pct(t and t["invalid_rate"]),
            f1cell(ts), pct(ts and ts["accuracy"]), pct(ts and ts["invalid_rate"]),
        ])
    seeds = [s for s in systems if re.fullmatch(r"qlora_seed\d+", s)]
    if len(seeds) > 1:
        for split in ("test", "test_timeshift"):
            vals = [metrics[f"{s}__{split}"]["macro_f1"] for s in seeds if f"{s}__{split}" in metrics]
            if len(vals) > 1:
                rows.append([f"QLoRA, mean ± sd over {len(vals)} seeds ({split})",
                             f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}", "", "", "", "", ""])
    head = ["System", "Macro-F1 test (95% CI)", "Accuracy", "Unparseable",
            "Macro-F1 time-shift (95% CI)", "Accuracy", "Unparseable"]
    return table(head, rows)


def section_paired(metrics: dict[str, dict]) -> str:
    from vclora.metrics import paired_bootstrap

    best_ft = "qlora_seed0" if (RES / "predictions" / "qlora_seed0__test.jsonl").exists() else None
    if not best_ft:
        return NOT_RUN
    rows = []
    for other in ["tfidf_logreg", "five_shot", "zero_shot", "qlora_seed0_w4a16"]:
        for split in ("test", "test_timeshift"):
            pa = RES / "predictions" / f"{best_ft}__{split}.jsonl"
            pb = RES / "predictions" / f"{other}__{split}.jsonl"
            if not (pa.exists() and pb.exists()):
                continue
            a = [json.loads(x) for x in pa.read_text().splitlines()]
            b = {r["odino"]: r for r in (json.loads(x) for x in pb.read_text().splitlines())}
            common = [r for r in a if r["odino"] in b]
            if len(common) < len(a):
                continue
            labels = metrics[f"{best_ft}__{split}"]["confusion_matrix"]["labels"]
            res = paired_bootstrap([r["label"] for r in common], [r["pred"] for r in common],
                                   [b[r["odino"]]["pred"] for r in common], labels)
            rows.append([f"{name(best_ft)} − {name(other)}", split, f"{res['diff']:+.3f}",
                         f"{res['ci95'][0]:+.3f} to {res['ci95'][1]:+.3f}"])
    if not rows:
        return NOT_RUN
    return table(["Comparison (macro-F1)", "Test set", "Difference", "Paired bootstrap 95% CI"], rows)


def section_per_class(metrics: dict[str, dict]) -> str:
    cols = [s for s in ["tfidf_logreg", "five_shot", "qlora_seed0", "qlora_seed0_w4a16"]
            if f"{s}__test" in metrics]
    if not cols:
        return NOT_RUN
    labels = metrics[f"{cols[0]}__test"]["confusion_matrix"]["labels"]
    rows = []
    for lab in labels:
        sup = metrics[f"{cols[0]}__test"]["per_class"][lab]["support"]
        rows.append([lab, str(sup)] + [f"{metrics[f'{c}__test']['per_class'][lab]['f1']:.3f}" for c in cols])
    return table(["Component", "Test support", *[name(c) for c in cols]], rows)


def section_training() -> str:
    runs = load("training/seed*.json")
    if not runs:
        return NOT_RUN
    rows = []
    for r in runs.values():
        gpu = (r["environment"].get("gpus") or ["CPU"])[0].split(",")[0]
        rows.append([str(r["seed"]), gpu, r["precision"], f"{r['train_examples']:,}", str(r["steps"]),
                     f"{r['train_seconds'] / 60:.1f}", f"{r['train_loss']:.4f}",
                     f"{r['trainable_params'] / 1e6:.1f} M of {r['total_params'] / 1e9:.2f} B",
                     "–" if r["peak_gpu_memory_gb"] is None else f"{r['peak_gpu_memory_gb']:.1f}"])
    return table(["Seed", "GPU", "Compute dtype", "Examples", "Steps", "Minutes", "Mean train loss",
                  "Trainable params", "Peak GPU mem (GiB)"], rows)


def prices() -> dict:
    return yaml.safe_load((ROOT / "bench" / "prices.yaml").read_text())


def cost_per_1k(rps: float, usd_per_hour: float | None) -> float | None:
    if not usd_per_hour or not rps:
        return None
    return usd_per_hour / (rps * 3600) * 1000


def section_offline(metrics: dict[str, dict]) -> str:
    runs = {k.removeprefix("offline_"): v for k, v in load("bench/offline_*.json").items()}
    if not runs:
        return NOT_RUN
    gpu_price = prices()["gpu"]
    rows = []
    for cfg in [c for c in OFFLINE_ORDER if c in runs] + [c for c in runs if c not in OFFLINE_ORDER]:
        r = runs[cfg]
        m = metrics.get(f"{CONFIG_SYSTEM.get(cfg, '')}__test")
        lat = r.get("batch_latency") or {}
        c = cost_per_1k(r["requests_per_second"], gpu_price["usd_per_hour"])
        mem = r.get("peak_gpu_memory_gb_torch") or r.get("peak_gpu_memory_gb_nvml")
        rows.append([OFFLINE_NAMES.get(cfg, cfg), f"{r['requests_per_second']:.1f}",
                     f"{r['output_tokens_per_second']:.0f}",
                     f"{lat['p50_ms']:.0f} / {lat['p95_ms']:.0f}" if lat else "–",
                     "–" if mem is None else f"{mem:.1f}",
                     "–" if c is None else f"${c:.4f}",
                     f"{m['macro_f1']:.3f}" if m else "–"])
    env = next(iter(runs.values()))["environment"]
    gpu = (env.get("gpus") or ["n/a"])[0]
    pk = env.get("packages", {})
    note = (f"\nGPU: {gpu}. torch {pk.get('torch')}, CUDA {env.get('cuda')}, transformers "
            f"{pk.get('transformers')}, vLLM {pk.get('vllm')}. {next(iter(runs.values()))['requests']} "
            "test prompts per configuration (256 at batch 1). Peak memory: torch allocator for HF; "
            "whole-device NVML reading for vLLM, which includes its pre-allocated KV cache. "
            f"Cost uses {gpu_price['usd_per_hour'] and '$' + str(gpu_price['usd_per_hour']) + '/hour'} "
            f"({gpu_price.get('name')}; source: {gpu_price.get('source')}).")
    return table(["Configuration", "Classifications/s", "Output tok/s", "Batch latency p50 / p95 (ms)",
                  "Peak GPU mem (GiB)", "Cost per 1,000", "Macro-F1 (test)"], rows) + "\n" + note


def section_serve() -> str:
    runs = {k.removeprefix("serve_"): v for k, v in load("bench/serve_*.json").items()}
    if not runs:
        return NOT_RUN
    rows = []
    for cfg, r in runs.items():
        for rate, x in r["rates"].items():
            rows.append([SERVE_NAMES.get(cfg, cfg), "unlimited" if rate == "inf" else rate,
                         f"{x['request_throughput']:.1f}", f"{x['output_throughput']:.0f}",
                         f"{x['median_ttft_ms']:.0f} / {x['p95_ttft_ms']:.0f}",
                         f"{x['median_itl_ms']:.1f} / {x['p95_itl_ms']:.1f}",
                         f"{x['median_e2el_ms']:.0f} / {x['p95_e2el_ms']:.0f}",
                         f"{x['completed']}/{x['num_prompts']}"])
    return table(["Server", "Request rate (req/s)", "Achieved req/s", "Output tok/s",
                  "TTFT p50 / p95 (ms)", "ITL p50 / p95 (ms)", "End-to-end p50 / p95 (ms)",
                  "Completed"], rows)


def section_figures() -> str:
    figs = [("throughput_vs_latency.png", "Throughput vs tail latency"), ("cost_vs_f1.png", "Cost vs F1")]
    have = [f"![{alt}](results/figures/{f})" for f, alt in figs if (RES / "figures" / f).exists()]
    return "\n".join(have) if have else NOT_RUN


def section_env() -> str:
    envs = []
    for pattern in ("metrics/*.json", "training/*.json", "bench/*.json"):
        for k, v in load(pattern).items():
            if "environment" in v:
                envs.append((pattern.split("/")[0] + "/" + k, v["environment"]))
    if not envs:
        return NOT_RUN
    seen, rows = set(), []
    for where, e in envs:
        key = (tuple(e.get("gpus") or []), e.get("cpu_count"), tuple(sorted(e.get("packages", {}).items())))
        if key in seen:
            continue
        seen.add(key)
        pk = e.get("packages", {})
        rows.append([", ".join(e.get("gpus") or []) or f"CPU only ({e.get('cpu_count')} cores)",
                     e.get("python", ""), ", ".join(f"{k} {v}" for k, v in sorted(pk.items())),
                     e.get("date_utc", ""), where])
    return table(["Hardware", "Python", "Packages", "Date", "First seen in"], rows)


# ---------------------------------------------------------------- charts

def charts(metrics: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = RES / "figures"
    fig_dir.mkdir(exist_ok=True)
    serve = {k.removeprefix("serve_"): v for k, v in load("bench/serve_*.json").items()}
    if serve:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        for cfg, r in serve.items():
            pts = sorted((x["request_throughput"], x["p95_e2el_ms"]) for x in r["rates"].values())
            ax.plot(*zip(*pts, strict=True), marker="o", label=SERVE_NAMES.get(cfg, cfg))
        ax.set_xlabel("Achieved throughput (classifications / s)")
        ax.set_ylabel("End-to-end latency p95 (ms)")
        ax.set_title("vLLM server: throughput vs tail latency")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "throughput_vs_latency.png")
        plt.close(fig)

    offline = {k.removeprefix("offline_"): v for k, v in load("bench/offline_*.json").items()}
    p = prices()
    pts = []
    for cfg, r in offline.items():
        m = metrics.get(f"{CONFIG_SYSTEM.get(cfg, '')}__test")
        c = cost_per_1k(r["requests_per_second"], p["gpu"]["usd_per_hour"])
        if m and c is not None:
            pts.append((c, m["macro_f1"], OFFLINE_NAMES.get(cfg, cfg)))
    tf = metrics.get("tfidf_logreg__test")
    c = tf and cost_per_1k(tf["classifications_per_second"], p["cpu"]["usd_per_hour"])
    if c:
        pts.append((c, tf["macro_f1"], "TF-IDF + LR (CPU)"))
    if pts:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        for c, f1, lab in pts:
            ax.scatter(c, f1, s=40)
            ax.annotate(lab, (c, f1), textcoords="offset points", xytext=(5, 4), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("Cost per 1,000 classifications (USD, log scale)")
        ax.set_ylabel("Macro-F1 on test")
        ax.set_title("Cost vs quality")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "cost_vs_f1.png")
        plt.close(fig)


def render_readme(text: str, sections: dict[str, str]) -> str:
    for key, body in sections.items():
        pat = re.compile(rf"(<!-- BEGIN:{key} -->\n).*?(\n<!-- END:{key} -->)", re.S)
        if not pat.search(text):
            sys.exit(f"README is missing the {key} markers")
        text = pat.sub(lambda m, b=body: m.group(1) + b + m.group(2), text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    metrics = load("metrics/*.json")
    if not args.check:
        charts(metrics)
    sections = {
        "data": section_data(),
        "results": section_results(metrics),
        "paired": section_paired(metrics),
        "per_class": section_per_class(metrics),
        "training": section_training(),
        "offline": section_offline(metrics),
        "serve": section_serve(),
        "env": section_env(),
        "figures": section_figures(),
    }
    old = README.read_text()
    new = render_readme(old, sections)
    if args.check:
        if new != old:
            sys.exit("README result sections are out of date: run `make report` and commit.")
        print("README is up to date with results/")
        return
    README.write_text(new)
    print("README and figures regenerated")


if __name__ == "__main__":
    main()
