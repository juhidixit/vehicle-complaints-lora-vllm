"""Reading the NHTSA ODI consumer-complaints flat file and turning it into labelled splits.

The flat file (FLAT_CMPL.txt inside FLAT_CMPL.zip) is tab-delimited with no header. Its field
order is documented in NHTSA's CMPL.txt layout file; ``verify_layout`` checks our field list
against that document before any parsing, so a layout change fails loudly instead of silently
shifting columns.

Important property of the file: one *line* is one (complaint, component) pair. A complaint
(``ODINO``) that names three components appears on three lines. We group lines by ODINO first.
"""

from __future__ import annotations

import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

FLAT_URL = "https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip"
LAYOUT_URL = "https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt"

# Field order of FLAT_CMPL.txt, as documented in CMPL.txt. The last two fields were added in 2026;
# older copies of the file have 49 fields per line, so both lengths are accepted.
FLAT_FIELDS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FAILDATE",
    "FIRE", "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES",
    "OCCURENCES", "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN",
    "ANTI_BRAKES_YN", "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE",
    "TRANS_TYPE", "VEH_SPEED", "DOT", "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE",
    "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE", "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL",
    "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP", "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN",
    "VEHICLES_TOWED_YN", "STATE_OF_INCIDENT", "VEHICLE_OPERATOR",
]
IDX = {name: i for i, name in enumerate(FLAT_FIELDS)}
ACCEPTED_FIELD_COUNTS = {49, 51}

# "UNKNOWN OR OTHER" is a frequent COMPDESC value but not a component; it is excluded as a target.
EXCLUDED_COMPONENTS = {"UNKNOWN OR OTHER"}
MIN_WORDS = 20


def verify_layout(layout_text: str) -> list[str]:
    """Check our field order against NHTSA's layout document (CMPL.txt).

    If the document numbers its fields ("1  CMPLID ..."), each documented number must match our
    position. Otherwise our field list must appear in the document as an ordered subsequence
    (names can also be mentioned earlier in prose, which a first-mention check would trip on).
    Returns the documented field names in our order; raises ``ValueError`` on a mismatch.
    """
    wanted = set(FLAT_FIELDS)
    numbered: dict[str, int] = {}
    for m in re.finditer(r"^\s*(\d{1,2})\s*[.:)]?\s+([A-Z][A-Z0-9_]+)\b", layout_text, re.M):
        if m.group(2) in wanted and m.group(2) not in numbered:
            numbered[m.group(2)] = int(m.group(1))
    for col in ("ODINO", "COMPDESC", "DATEA", "CDESCR"):
        if col not in layout_text:
            raise ValueError(f"layout document does not mention {col}; check the NHTSA layout")
    if len(numbered) >= 20:
        wrong = {k: v for k, v in numbered.items() if v != IDX[k] + 1}
        if wrong:
            raise ValueError(f"field numbers differ from ours (doc number vs ours): "
                             f"{ {k: (v, IDX[k] + 1) for k, v in wrong.items()} }")
        return [f for f in FLAT_FIELDS if f in numbered]
    tokens = [t for t in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", layout_text) if t in wanted]
    it = iter(tokens)
    documented = [f for f in FLAT_FIELDS if f in tokens]
    if not all(any(t == f for t in it) for f in documented):
        raise ValueError(f"field order differs from the layout document: {tokens}")
    return documented


def top_level(compdesc: str) -> str:
    """``'ELECTRICAL SYSTEM:12V/24V/48V BATTERY'`` -> ``'ELECTRICAL SYSTEM'``."""
    return compdesc.split(":", 1)[0].strip().upper()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def iter_flat_lines(path: Path) -> Iterator[str]:
    """Yield decoded lines from FLAT_CMPL.zip (or an already extracted .txt)."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            member = next(n for n in zf.namelist() if n.upper().endswith(".TXT"))
            with zf.open(member) as fh:
                yield from io.TextIOWrapper(fh, encoding="latin-1", newline="\n")
    else:
        with open(path, encoding="latin-1", newline="\n") as fh:
            yield from fh


@dataclass
class Complaint:
    odino: str
    datea: str  # YYYYMMDD, date NHTSA received the complaint
    text: str
    make: str = ""
    model_year: str = ""
    components: set[str] = field(default_factory=set)  # full COMPDESC values

    @property
    def top_levels(self) -> set[str]:
        return {top_level(c) for c in self.components}


def group_complaints(
    lines: Iterable[str], min_date: str, max_date: str, stats: Counter
) -> dict[str, Complaint]:
    """Group flat-file lines into complaints, keeping complaints received in [min_date, max_date]."""
    out: dict[str, Complaint] = {}
    for raw in lines:
        stats["lines"] += 1
        parts = raw.rstrip("\r\n").split("\t")
        if len(parts) not in ACCEPTED_FIELD_COUNTS:
            stats["lines_bad_field_count"] += 1
            continue
        odino, datea = parts[IDX["ODINO"]].strip(), parts[IDX["DATEA"]].strip()
        if not odino.isdigit() or not re.fullmatch(r"\d{8}", datea):
            stats["lines_bad_id_or_date"] += 1
            continue
        if not (min_date <= datea <= max_date):
            continue
        comp = parts[IDX["COMPDESC"]].strip().upper()
        c = out.get(odino)
        if c is None:
            c = out[odino] = Complaint(
                odino=odino,
                datea=datea,
                text=normalize_text(parts[IDX["CDESCR"]]),
                make=parts[IDX["MAKETXT"]].strip(),
                model_year=parts[IDX["YEARTXT"]].strip(),
            )
        if comp:
            c.components.add(comp)
    stats["complaints_in_date_range"] = len(out)
    return out


def filter_complaints(complaints: Iterable[Complaint], stats: Counter, prefix: str) -> list[dict]:
    """Single top-level component, >= MIN_WORDS words, exact-text de-duplication.

    Narratives that occur more than once are dropped entirely when the copies disagree on the
    label, otherwise the copy with the lowest ODINO is kept.
    """
    rows = []
    for c in complaints:
        tl = c.top_levels - EXCLUDED_COMPONENTS
        if len(c.top_levels) != 1 or not tl:
            stats[f"{prefix}_dropped_multi_or_unknown_component"] += 1
            continue
        if len(c.text.split()) < MIN_WORDS:
            stats[f"{prefix}_dropped_short"] += 1
            continue
        rows.append(
            {"odino": c.odino, "datea": c.datea, "label": next(iter(tl)), "text": c.text,
             "make": c.make, "model_year": c.model_year}
        )
    by_text: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_text[r["text"].lower()].append(r)
    kept = []
    for group in by_text.values():
        if len({r["label"] for r in group}) > 1:
            stats[f"{prefix}_dropped_duplicate_text_conflicting_label"] += len(group)
            continue
        group.sort(key=lambda r: int(r["odino"]))
        kept.append(group[0])
        stats[f"{prefix}_dropped_duplicate_text"] += len(group) - 1
    kept.sort(key=lambda r: int(r["odino"]))
    stats[f"{prefix}_kept"] = len(kept)
    return kept


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n rows keeping label proportions (largest-remainder allocation)."""
    if n >= len(rows):
        return list(rows)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    total = len(rows)
    quotas = {lab: n * len(v) / total for lab, v in by_label.items()}
    alloc = {lab: int(q) for lab, q in quotas.items()}
    for lab in sorted(quotas, key=lambda k: (alloc[k] - quotas[k], k))[: n - sum(alloc.values())]:
        alloc[lab] += 1
    rng = random.Random(seed)
    out = []
    for lab in sorted(by_label):
        out.extend(rng.sample(by_label[lab], alloc[lab]))
    rng.shuffle(out)
    return out


def stratified_split(rows: list[dict], sizes: dict[str, int], seed: int) -> dict[str, list[dict]]:
    """Split rows (already sampled to sum(sizes)) into stratified parts, in the order given."""
    remaining = list(rows)
    out = {}
    names = list(sizes)
    for i, name in enumerate(names):
        if i == len(names) - 1:
            out[name] = remaining
            break
        part = stratified_sample(remaining, sizes[name], seed + i)
        ids = {r["odino"] for r in part}
        remaining = [r for r in remaining if r["odino"] not in ids]
        out[name] = part
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
