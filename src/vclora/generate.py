"""Greedy generation with either Hugging Face Transformers or vLLM, behind one interface."""

from __future__ import annotations

import time
from dataclasses import dataclass

MAX_NEW_TOKENS = 16  # the longest label is about 8 tokens


def render(tokenizer, messages_list: list[list[dict]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]


@dataclass
class GenResult:
    texts: list[str]
    seconds: float
    output_tokens: int
    prompt_tokens: int


class HFGenerator:
    def __init__(self, model_path: str, adapter: str | None = None, dtype: str = "auto",
                 batch_size: int = 16, device: str | None = None, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(adapter or model_path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = {"dtype": resolve_dtype(dtype)}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=resolve_dtype(dtype))
            kwargs["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        if not load_in_4bit:
            model = model.to(self.device)
        self.model = model.eval()
        self.batch_size = batch_size

    def generate(self, prompts: list[str], batch_size: int | None = None) -> GenResult:
        bs = batch_size or self.batch_size
        torch = self.torch
        texts, out_tok, in_tok = [], 0, 0
        t0 = time.perf_counter()
        for i in range(0, len(prompts), bs):
            enc = self.tok(prompts[i : i + bs], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with torch.inference_mode():
                out = self.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            for row in new:
                ids = [t for t in row.tolist() if t != self.tok.pad_token_id]
                out_tok += len(ids)
                texts.append(self.tok.decode(ids, skip_special_tokens=True))
            in_tok += int(enc["attention_mask"].sum())
        if self.device == "cuda":
            torch.cuda.synchronize()
        return GenResult(texts, time.perf_counter() - t0, out_tok, in_tok)


class VLLMGenerator:
    def __init__(self, model_path: str, adapter: str | None = None, dtype: str = "auto",
                 max_model_len: int = 1024, gpu_memory_utilization: float = 0.85, **kwargs):
        from vllm import LLM, SamplingParams

        self.adapter = adapter
        if adapter:
            kwargs.update(enable_lora=True, max_lora_rank=16)
        self.llm = LLM(model=model_path, dtype=dtype, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, seed=0, **kwargs)
        self.params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
        self.tok = self.llm.get_tokenizer()

    def generate(self, prompts: list[str], batch_size: int | None = None) -> GenResult:
        lora = None
        if self.adapter:
            from vllm.lora.request import LoRARequest

            lora = LoRARequest("ft", 1, self.adapter)
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, self.params, lora_request=lora, use_tqdm=False)
        secs = time.perf_counter() - t0
        return GenResult(
            [o.outputs[0].text for o in outs], secs,
            sum(len(o.outputs[0].token_ids) for o in outs),
            sum(len(o.prompt_token_ids) for o in outs),
        )


def resolve_dtype(dtype: str):
    import torch

    if dtype == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if has_native_bf16() else torch.float16
        return torch.float32
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
            "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}[dtype]


def vllm_dtype() -> str:
    """bf16 where the GPU supports it, fp16 otherwise (T4 and older have no bf16)."""
    return "bfloat16" if has_native_bf16() else "float16"


def has_native_bf16() -> bool:
    """Ampere (compute capability 8.0) or newer. T4 is 7.5: torch may emulate bf16 there, slowly."""
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
