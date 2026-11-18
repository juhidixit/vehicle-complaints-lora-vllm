from sklearn.metrics import accuracy_score, f1_score

from vclora.metrics import paired_bootstrap, score
from vclora.prompting import INVALID

LABELS = ["A", "B", "C"]


def test_score_matches_sklearn_and_counts_invalid_as_wrong():
    y = ["A", "A", "B", "B", "C", "C", "A", "B"]
    p = ["A", "B", "B", INVALID, "C", "A", "A", "B"]
    m = score(y, p, LABELS)
    assert abs(m["macro_f1"] - f1_score(y, p, labels=LABELS, average="macro")) < 1e-12
    assert abs(m["accuracy"] - accuracy_score(y, p)) < 1e-12
    assert m["invalid_rate"] == 1 / 8
    assert m["confusion_matrix"]["matrix"][1][3] == 1  # B predicted INVALID
    lo, hi = m["macro_f1_ci95"]
    assert lo <= m["macro_f1"] <= hi


def test_paired_bootstrap_sign():
    y = ["A", "B", "C"] * 50
    good = list(y)
    bad = ["A"] * len(y)
    r = paired_bootstrap(y, good, bad, LABELS)
    assert r["diff"] > 0 and r["ci95"][0] > 0
