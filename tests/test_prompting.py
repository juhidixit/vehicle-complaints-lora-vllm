from vclora.prompting import INVALID, build_messages, parse_label, pick_shots

LABELS = ["ENGINE", "ENGINE AND ENGINE COOLING", "SERVICE BRAKES", "STEERING", "POWER TRAIN"]


def test_parse_exact_and_case():
    assert parse_label("SERVICE BRAKES", LABELS) == "SERVICE BRAKES"
    assert parse_label("  service brakes.\n", LABELS) == "SERVICE BRAKES"
    assert parse_label("Steering\nbecause the wheel...", LABELS) == "STEERING"


def test_parse_overlapping_labels_takes_longest_match():
    assert parse_label("ENGINE AND ENGINE COOLING", LABELS) == "ENGINE AND ENGINE COOLING"
    assert parse_label("ENGINE", LABELS) == "ENGINE"


def test_parse_prefix_and_invalid():
    assert parse_label("POWER", LABELS) == "POWER TRAIN"
    assert parse_label("", LABELS) == INVALID
    assert parse_label("I think it is the brakes", LABELS) == INVALID
    assert parse_label("TIRES", LABELS) == INVALID


def test_messages_and_shots():
    train = [{"odino": str(i), "label": LABELS[i % 5], "text": f"text {i} " * 200} for i in range(50)]
    shots = pick_shots(train, 5, seed=7)
    assert len({s["label"] for s in shots}) == 5
    assert pick_shots(train, 5, seed=7) == shots
    msgs = build_messages("word " * 1000, LABELS, shots)
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 5 + ["user"]
    assert all(lab in msgs[0]["content"] for lab in LABELS)
    assert len(msgs[-1]["content"].split()) < 320  # narrative truncated
