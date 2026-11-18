"""Build train / val / test / time-shift splits from the NHTSA complaints flat file.

    python data/build_splits.py --flat data/raw/FLAT_CMPL.zip            # fresh splits
    python data/build_splits.py --flat data/raw/FLAT_CMPL.zip --from-ids # rebuild committed splits

Writes the full examples to data/processed/*.jsonl (git-ignored), the complaint IDs of every
split to data/splits/*.csv (committed) and the data-card statistics to results/data_stats.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

from vclora.nhtsa import (
    EXCLUDED_COMPONENTS,
    MIN_WORDS,
    filter_complaints,
    group_complaints,
    iter_flat_lines,
    stratified_sample,
    stratified_split,
    write_jsonl,
)

SIZES = {"train": 20_000, "val": 2_000, "test": 2_000}
TIMESHIFT_SIZE = 2_000
POOL_YEARS = 10
N_LABELS = 10


def latest_full_year(max_date: str) -> int:
    year = int(max_date[:4])
    return year if max_date[4:] >= "1231" else year - 1


def split_stats(rows: list[dict]) -> dict:
    words = [len(r["text"].split()) for r in rows]
    years = Counter(r["datea"][:4] for r in rows)
    return {
        "n": len(rows),
        "labels": dict(sorted(Counter(r["label"] for r in rows).items(), key=lambda kv: -kv[1])),
        "words_median": statistics.median(words) if words else 0,
        "words_p95": sorted(words)[int(0.95 * (len(words) - 1))] if words else 0,
        "received_years": dict(sorted(years.items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--stats", type=Path, default=Path("results/data_stats.json"))
    ap.add_argument("--download-meta", type=Path, default=Path("data/raw/download_meta.json"))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--from-ids", action="store_true", help="rebuild the committed splits")
    ap.add_argument("--sizes", type=int, nargs=4, metavar=("TRAIN", "VAL", "TEST", "SHIFT"),
                    default=[*SIZES.values(), TIMESHIFT_SIZE], help="split sizes (tests use small ones)")
    args = ap.parse_args()

    sizes = dict(zip(SIZES, args.sizes[:3], strict=True))
    timeshift_size = args.sizes[3]
    stats: Counter = Counter()
    complaints = group_complaints(iter_flat_lines(args.flat), "20100101", "29991231", stats)  # pool starts later
    max_date = max(c.datea for c in complaints.values())
    full_year = latest_full_year(max_date)
    pool_lo, pool_hi = full_year - POOL_YEARS, full_year - 1
    print(f"latest received date {max_date}; time-shift year {full_year}; pool {pool_lo}-{pool_hi}")

    pool_c = [c for c in complaints.values() if pool_lo <= int(c.datea[:4]) <= pool_hi]
    shift_c = [c for c in complaints.values() if int(c.datea[:4]) == full_year]
    pool = filter_complaints(pool_c, stats, "pool")
    shift = filter_complaints(shift_c, stats, "timeshift")

    if args.from_ids:
        by_id = {r["odino"]: r for r in pool + shift}
        splits = {}
        for name in [*sizes, "test_timeshift"]:
            with open(args.splits_dir / f"{name}.csv", newline="") as fh:
                ids = [(row["odino"], row["label"]) for row in csv.DictReader(fh)]
            rows = [by_id[i] for i, lab in ids if i in by_id and by_id[i]["label"] == lab]
            missing = len(ids) - len(rows)
            print(f"{name}: {len(rows)} rebuilt, {missing} missing or relabelled")
            if missing > 0.01 * len(ids):
                sys.exit(f"{name}: more than 1% of committed IDs are missing from this file version")
            splits[name] = rows
        labels = sorted({r["label"] for r in splits["train"]})
    else:
        counts = Counter(r["label"] for r in pool)
        labels = sorted(lab for lab, _ in counts.most_common(N_LABELS))
        pool_top = [r for r in pool if r["label"] in labels]
        sampled = stratified_sample(pool_top, sum(sizes.values()), args.seed)
        splits = stratified_split(sampled, sizes, args.seed)
        shift_top = [r for r in shift if r["label"] in labels]
        rng = random.Random(args.seed)
        splits["test_timeshift"] = rng.sample(shift_top, min(timeshift_size, len(shift_top)))
        stats["pool_top10_available"] = len(pool_top)
        stats["timeshift_top10_available"] = len(shift_top)

    all_ids = [r["odino"] for rows in splits.values() for r in rows]
    assert len(all_ids) == len(set(all_ids)), "a complaint appears in two splits"

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(args.out_dir / f"{name}.jsonl", rows)
        if not args.from_ids:
            with open(args.splits_dir / f"{name}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["odino", "label"])
                w.writerows((r["odino"], r["label"]) for r in rows)
    (args.out_dir / "labels.json").write_text(json.dumps(labels, indent=1))

    meta = json.loads(args.download_meta.read_text()) if args.download_meta.exists() else {}
    out = {
        "source": meta,
        "latest_received_date": max_date,
        "pool_years": [pool_lo, pool_hi],
        "timeshift_year": full_year,
        "labels": labels,
        "excluded_components": sorted(EXCLUDED_COMPONENTS),
        "min_words": MIN_WORDS,
        "seed": args.seed,
        "funnel": dict(stats),
        "splits": {name: split_stats(rows) for name, rows in splits.items()},
    }
    if not args.from_ids:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["funnel"], indent=1))


if __name__ == "__main__":
    main()
