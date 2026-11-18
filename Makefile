# Each target writes JSON under results/; `make report` turns those into the README tables.
PY ?= python
BASE ?= Qwen/Qwen2.5-1.5B-Instruct
SEEDS ?= 0
SEED0 := $(firstword $(SEEDS))

.PHONY: all data data-from-ids baselines llm-baselines train eval quantize bench report test lint

all: data-from-ids baselines llm-baselines train eval quantize bench report

data:
	$(PY) data/download.py --out data/raw
	$(PY) data/build_splits.py --flat data/raw/FLAT_CMPL.zip

data-from-ids:
	$(PY) data/download.py --out data/raw --skip-if-present
	$(PY) data/build_splits.py --flat data/raw/FLAT_CMPL.zip --from-ids

baselines:
	$(PY) eval/baselines_classic.py

llm-baselines:
	$(PY) eval/llm_eval.py --system zero_shot --model $(BASE)
	$(PY) eval/llm_eval.py --system five_shot --model $(BASE) --shots 5

train:
	for s in $(SEEDS); do $(PY) train/qlora.py --seed $$s || exit 1; done

eval:
	for s in $(SEEDS); do \
	  $(PY) train/merge.py --adapter outputs/adapters/seed$$s --out outputs/merged/seed$$s && \
	  $(PY) eval/llm_eval.py --system qlora_seed$$s --model outputs/merged/seed$$s --family finetuned || exit 1; \
	done

QVENV ?= .venv-quant
quantize:
	test -x $(QVENV)/bin/python || ($(PY) -m venv $(QVENV) && $(QVENV)/bin/pip install -q -e ".[quant]")
	$(QVENV)/bin/python bench/quantize.py --model outputs/merged/seed$(SEED0) --out outputs/w4a16/seed$(SEED0)
	$(PY) eval/llm_eval.py --system qlora_seed$(SEED0)_w4a16 --model outputs/w4a16/seed$(SEED0) --family finetuned

bench:  # each configuration is independent: a failure (e.g. a kernel the GPU lacks) is logged, the rest still run
	-for c in hf_bs1 hf_bs8 hf_bs32 vllm; do \
	  $(PY) bench/offline.py --config $$c --model outputs/merged/seed$(SEED0); done
	-$(PY) bench/offline.py --config vllm_w4a16 --model outputs/w4a16/seed$(SEED0)
	-$(PY) bench/offline.py --config vllm_lora --model $(BASE) --adapter outputs/adapters/seed$(SEED0)
	-$(PY) bench/serve_load.py --config merged --model outputs/merged/seed$(SEED0)
	-$(PY) bench/serve_load.py --config w4a16 --model outputs/w4a16/seed$(SEED0)
	-$(PY) bench/serve_load.py --config lora --model $(BASE) --adapter outputs/adapters/seed$(SEED0)

report:
	$(PY) report/make_report.py

test:
	$(PY) -m pytest -q

lint:
	ruff check .
