"""
AutoNerve LLM layer — local, in-process, via HuggingFace transformers.
Same approach as the UPS classifier (no Ollama server). Reuses Qwen2.5-1.5B-Instruct.

The LLM only EXTRACTS and EXPLAINS — it never computes. engine.py does the numbers.
If the model folder is absent, callers fall back to a deterministic keyword
extractor (extraction.py), so the demo never breaks.
"""
from __future__ import annotations
import os
import re
import json
from pathlib import Path

HERE = Path(__file__).parent
# reuse the same local model you already have for the UPS app
MODEL_DIR = os.environ.get("AUTONERVE_MODEL", str(HERE / "models" / "Qwen2.5-1.5B-Instruct"))

_tok = None
_model = None
_device = None


def available() -> bool:
    return Path(MODEL_DIR).exists()


def _load():
    global _tok, _model, _device
    if _model is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    dtype = torch.float16 if _device == "cuda" else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(_device)
    _model.eval()


def generate_text(prompt: str, system: str = "You are a precise assistant.",
                  max_new_tokens: int = 256) -> str:
    """Greedy, deterministic generation. No sampling kwargs (matches do_sample=False)."""
    import torch
    _load()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    text = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tok(text, return_tensors="pt", truncation=True, max_length=4096).to(_device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=_tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[-1]:]
    return _tok.decode(gen, skip_special_tokens=True).strip()


def _first_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON in output: {text[:160]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError(f"incomplete JSON: {text[:160]}")


def generate_json(prompt: str, system: str = "You return STRICT JSON only. No prose.",
                  max_new_tokens: int = 256) -> dict:
    raw = generate_text(prompt, system=system, max_new_tokens=max_new_tokens)
    return json.loads(_first_json(raw))
