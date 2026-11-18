"""Fixtures: a synthetic NHTSA flat file, synthetic splits, and a tiny random Qwen2 model.

Everything is built locally, so the tests need no network and no GPU.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from vclora.nhtsa import IDX

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["AIR BAGS", "ELECTRICAL SYSTEM", "ENGINE", "SERVICE BRAKES", "STEERING"]
WORDS = {
    "AIR BAGS": "airbag light deployed crash sensor passenger seat warning",
    "ELECTRICAL SYSTEM": "battery dead dashboard lights flicker alternator wiring fuse",
    "ENGINE": "engine stalled check light oil misfire coolant overheating",
    "SERVICE BRAKES": "brakes pedal soft grinding stopping distance rotor abs",
    "STEERING": "steering wheel hard turn power assist loss pulling column",
}
FILLER = "the vehicle was driving on the highway when suddenly the dealer said it was fine".split()


def make_text(label: str, rng: random.Random, n: int = 30) -> str:
    vocab = WORDS[label].split() + FILLER
    return " ".join(rng.choice(vocab) for _ in range(n))


def flat_line(odino: int, comp: str, datea: str, text: str, n_fields: int = 51) -> str:
    f = [""] * n_fields
    f[IDX["CMPLID"]] = str(odino * 10)
    f[IDX["ODINO"]] = str(odino)
    f[IDX["MAKETXT"]] = "ACME"
    f[IDX["YEARTXT"]] = "2020"
    f[IDX["COMPDESC"]] = comp
    f[IDX["DATEA"]] = datea
    f[IDX["CDESCR"]] = text
    return "\t".join(f)


@pytest.fixture(scope="session")
def flat_zip(tmp_path_factory) -> Path:
    """~1,700 synthetic complaints over 2014-2026 with the edge cases the filters must handle."""
    rng = random.Random(0)
    lines, odino = [], 10_000_000
    for year in range(2014, 2027):
        for _ in range(130):
            odino += 1
            lab = rng.choice(LABELS)
            datea = f"{year}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
            if year == 2026:
                datea = f"2026{rng.randint(1, 10):02d}{rng.randint(1, 28):02d}"
            text = make_text(lab, rng)
            lines.append(flat_line(odino, f"{lab}:SUBPART {rng.randint(1, 3)}", datea, text,
                                   rng.choice([49, 51])))
            if rng.random() < 0.1:  # same complaint, second sub-component of the same top level
                lines.append(flat_line(odino, f"{lab}:OTHER PART", datea, text))
    # edge cases, all in 2020
    lines.append(flat_line(1, "ENGINE", "20200101", "too short to keep"))
    lines.append(flat_line(2, "ENGINE", "20200101", make_text("ENGINE", rng)))
    lines.append(flat_line(2, "STEERING", "20200101", make_text("ENGINE", rng)))  # multi-component
    lines.append(flat_line(3, "UNKNOWN OR OTHER", "20200101", make_text("ENGINE", rng)))
    dup = make_text("STEERING", rng)
    lines.append(flat_line(4, "STEERING", "20200102", dup))
    lines.append(flat_line(5, "STEERING", "20200103", dup))  # exact duplicate text
    lines.append("garbage line with no tabs")
    path = tmp_path_factory.mktemp("raw") / "FLAT_CMPL.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("FLAT_CMPL.txt", "\n".join(lines) + "\n")
    return path


@pytest.fixture(scope="session")
def workspace(tmp_path_factory, flat_zip) -> dict:
    """Run data/build_splits.py on the synthetic file; returns env vars pointing at the outputs."""
    ws = tmp_path_factory.mktemp("ws")
    env = {**os.environ, "VCL_DATA": str(ws / "processed"), "VCL_RESULTS": str(ws / "results")}
    cmd = [sys.executable, str(ROOT / "data/build_splits.py"), "--flat", str(flat_zip),
           "--out-dir", str(ws / "processed"), "--splits-dir", str(ws / "splits"),
           "--stats", str(ws / "results/data_stats.json"), "--sizes", "600", "100", "100", "80"]
    subprocess.run(cmd, check=True, env=env, cwd=ws)
    return {"env": env, "ws": ws, "flat": flat_zip}


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> Path:
    """A 2-layer random Qwen2 model with a small BPE tokenizer and a ChatML chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out = tmp_path_factory.mktemp("tiny")
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = [" ".join(WORDS.values()), " ".join(LABELS), " ".join(FILLER),
              "You classify US vehicle safety complaints Complaint Category system user assistant"]
    tk.train_from_iterator(corpus * 20, trainers.BpeTrainer(
        vocab_size=400, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    from tokenizers import decoders

    tk.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    tok.save_pretrained(out)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                      bos_token_id=tok.eos_token_id, tie_word_embeddings=True)
    import torch

    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(out)
    (out / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id, "do_sample": False}))
    return out
