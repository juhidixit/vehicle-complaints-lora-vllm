import csv
import json
import subprocess
import sys
from collections import Counter

import pytest

from vclora.nhtsa import (
    FLAT_FIELDS,
    stratified_sample,
    stratified_split,
    top_level,
    verify_layout,
)

from .conftest import ROOT


def test_top_level():
    assert top_level("ELECTRICAL SYSTEM:12V/24V/48V BATTERY") == "ELECTRICAL SYSTEM"
    assert top_level("air bags") == "AIR BAGS"


def test_verify_layout_accepts_documented_order_and_rejects_swaps():
    doc = "\n".join(f"{i + 1:>3} {name:<20} CHAR(10)  description" for i, name in enumerate(FLAT_FIELDS))
    assert verify_layout(doc) == FLAT_FIELDS
    swapped = doc.replace("COMPDESC", "TMP").replace("CDESCR", "COMPDESC").replace("TMP", "CDESCR")
    with pytest.raises(ValueError):
        verify_layout(swapped)


def test_stratified_sample_keeps_proportions():
    rows = [{"odino": str(i), "label": "A" if i % 4 else "B"} for i in range(1000)]
    s = stratified_sample(rows, 100, seed=1)
    assert Counter(r["label"] for r in s) == {"A": 75, "B": 25}
    parts = stratified_split(s, {"train": 60, "val": 20, "test": 20}, seed=1)
    assert [len(p) for p in parts.values()] == [60, 20, 20]
    ids = [r["odino"] for p in parts.values() for r in p]
    assert len(ids) == len(set(ids))


def test_build_splits(workspace):
    ws = workspace["ws"]
    stats = json.loads((ws / "results/data_stats.json").read_text())
    assert stats["timeshift_year"] == 2025
    assert stats["pool_years"] == [2015, 2024]
    f = stats["funnel"]
    assert f["lines_bad_field_count"] == 1
    assert f["pool_dropped_short"] >= 1
    assert f["pool_dropped_multi_or_unknown_component"] >= 2  # ODINO 2 and 3
    assert f["pool_dropped_duplicate_text"] >= 1  # ODINO 5
    sizes = {k: v["n"] for k, v in stats["splits"].items()}
    assert sizes == {"train": 600, "val": 100, "test": 100, "test_timeshift": 80}
    for name in ("train", "test_timeshift"):
        rows = [json.loads(x) for x in (ws / "processed" / f"{name}.jsonl").read_text().splitlines()]
        years = {r["datea"][:4] for r in rows}
        assert years <= ({"2025"} if name == "test_timeshift" else {str(y) for y in range(2015, 2025)})
        assert all(len(r["text"].split()) >= 20 for r in rows)
    ids = []
    for name in ("train", "val", "test", "test_timeshift"):
        with open(ws / "splits" / f"{name}.csv") as fh:
            ids += [r["odino"] for r in csv.DictReader(fh)]
    assert len(ids) == len(set(ids)) == 880
    assert not {"2", "3", "5"} & set(ids)


def test_rebuild_from_ids_is_identical(workspace, tmp_path):
    ws, env = workspace["ws"], workspace["env"]
    out = tmp_path / "rebuilt"
    subprocess.run([sys.executable, str(ROOT / "data/build_splits.py"), "--flat", str(workspace["flat"]),
                    "--out-dir", str(out), "--splits-dir", str(ws / "splits"), "--from-ids"],
                   check=True, env=env, cwd=tmp_path)
    for name in ("train", "val", "test", "test_timeshift"):
        assert (out / f"{name}.jsonl").read_text() == (ws / "processed" / f"{name}.jsonl").read_text()
