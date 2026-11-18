"""CPU baselines: majority class and TF-IDF + logistic regression.

    python eval/baselines_classic.py

The logistic-regression settings are chosen on the validation split only (macro-F1), then the
chosen model is scored once on each test split.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from vclora.nhtsa import read_jsonl
from vclora.results import DATA, SPLITS, environment, load_labels, save_run

GRID = {"C": [1.0, 4.0, 16.0], "class_weight": [None, "balanced"]}


def main() -> None:
    labels = load_labels()
    train, val = read_jsonl(DATA / "train.jsonl"), read_jsonl(DATA / "val.jsonl")
    tests = {s: read_jsonl(DATA / f"{s}.jsonl") for s in SPLITS}
    env = environment()

    majority = Counter(r["label"] for r in train).most_common(1)[0][0]
    for split, rows in tests.items():
        save_run("majority", split, rows, [majority] * len(rows), labels,
                 extra={"family": "baseline", "predicts": majority, "environment": env})

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    t0 = time.perf_counter()
    x_train = vec.fit_transform(r["text"] for r in train)
    x_val = vec.transform(r["text"] for r in val)
    y_train, y_val = [r["label"] for r in train], [r["label"] for r in val]

    search = []
    for c, cw in itertools.product(GRID["C"], GRID["class_weight"]):
        clf = LogisticRegression(C=c, class_weight=cw, max_iter=2000)
        clf.fit(x_train, y_train)
        f1 = f1_score(y_val, clf.predict(x_val), labels=labels, average="macro")
        search.append({"C": c, "class_weight": cw, "val_macro_f1": f1})
        print(f"C={c:<5} class_weight={cw!s:9s} val macro-F1 {f1:.4f}")
    best = max(search, key=lambda s: s["val_macro_f1"])
    clf = LogisticRegression(C=best["C"], class_weight=best["class_weight"], max_iter=2000)
    clf.fit(x_train, y_train)
    fit_s = time.perf_counter() - t0

    for split, rows in tests.items():
        texts = [r["text"] for r in rows]
        t1 = time.perf_counter()
        preds = list(clf.predict(vec.transform(texts)))
        infer_s = time.perf_counter() - t1
        save_run("tfidf_logreg", split, rows, preds, labels, extra={
            "family": "baseline",
            "params": {"C": best["C"], "class_weight": best["class_weight"], "ngram_range": [1, 2],
                       "min_df": 2, "max_features": 200_000},
            "val_search": search,
            "fit_seconds_incl_search": fit_s,
            "inference_seconds": infer_s,
            "classifications_per_second": len(rows) / infer_s,
            "environment": env,
        })


if __name__ == "__main__":
    main()
