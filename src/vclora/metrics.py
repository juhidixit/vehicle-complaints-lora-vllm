"""Classification metrics with bootstrap confidence intervals.

Predictions outside the label list (unparseable LLM output) count as wrong: they are a false
negative for the true class and a false positive for no class.
"""

from __future__ import annotations

import numpy as np

from vclora.prompting import INVALID

N_BOOT = 1000


def _encode(y_true, y_pred, labels):
    idx = {lab: i for i, lab in enumerate(labels)}
    t = np.array([idx[y] for y in y_true], dtype=np.int64)
    p = np.array([idx.get(y, len(labels)) for y in y_pred], dtype=np.int64)  # len(labels) = invalid
    return t, p


def _f1_per_class(t: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    tp = np.bincount(t[t == p], minlength=k)[:k].astype(float)
    pred_cnt = np.bincount(p, minlength=k + 1)[:k].astype(float)
    true_cnt = np.bincount(t, minlength=k)[:k].astype(float)
    denom = pred_cnt + true_cnt
    return np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)


def macro_f1(t: np.ndarray, p: np.ndarray, k: int) -> float:
    return float(_f1_per_class(t, p, k).mean())


def _ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def score(y_true: list[str], y_pred: list[str], labels: list[str], seed: int = 0) -> dict:
    k = len(labels)
    t, p = _encode(y_true, y_pred, labels)
    n = len(t)
    f1 = _f1_per_class(t, p, k)
    true_cnt = np.bincount(t, minlength=k)
    pred_cnt = np.bincount(p, minlength=k + 1)
    tp = np.bincount(t[t == p], minlength=k)[:k]
    cm = np.zeros((k, k + 1), dtype=np.int64)
    np.add.at(cm, (t, p), 1)

    rng = np.random.default_rng(seed)
    boot_f1, boot_acc = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        boot_f1[b] = macro_f1(t[i], p[i], k)
        boot_acc[b] = float((t[i] == p[i]).mean())

    return {
        "n": n,
        "macro_f1": float(f1.mean()),
        "macro_f1_ci95": _ci(boot_f1),
        "accuracy": float((t == p).mean()),
        "accuracy_ci95": _ci(boot_acc),
        "invalid_rate": float((p == k).mean()),
        "per_class": {
            lab: {
                "f1": float(f1[j]),
                "precision": float(tp[j] / pred_cnt[j]) if pred_cnt[j] else 0.0,
                "recall": float(tp[j] / true_cnt[j]) if true_cnt[j] else 0.0,
                "support": int(true_cnt[j]),
            }
            for j, lab in enumerate(labels)
        },
        "confusion_matrix": {"labels": labels, "columns": [*labels, INVALID], "matrix": cm.tolist()},
    }


def paired_bootstrap(
    y_true: list[str], pred_a: list[str], pred_b: list[str], labels: list[str], seed: int = 0
) -> dict:
    """Macro-F1 difference (a - b) with a paired bootstrap CI over the same test examples."""
    k = len(labels)
    t, pa = _encode(y_true, pred_a, labels)
    _, pb = _encode(y_true, pred_b, labels)
    n = len(t)
    rng = np.random.default_rng(seed)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        diffs[b] = macro_f1(t[i], pa[i], k) - macro_f1(t[i], pb[i], k)
    return {
        "diff": macro_f1(t, pa, k) - macro_f1(t, pb, k),
        "ci95": _ci(diffs),
        "p_a_not_better": float((diffs <= 0).mean()),
    }
