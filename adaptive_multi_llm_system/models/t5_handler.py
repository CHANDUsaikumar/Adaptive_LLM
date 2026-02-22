"""T5 summarization integration using Hugging Face Transformers.

Provides:
- generate_summary(text: str) -> dict: uses 't5-small' to produce a summary
- T5Handler.summarize(texts: List[str]) -> List[str] for compatibility

If transformers/torch are unavailable, a lightweight fallback is used.
"""
from typing import List, Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _HF_AVAILABLE = True
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    _HF_AVAILABLE = False

MODEL_NAME = "t5-small"

# module-level tokenizer and model
_tokenizer = None
_model = None
_device = "cpu"


def _load_model_if_needed():
    global _tokenizer, _model, _device
    if not _HF_AVAILABLE:
        return
    if _model is None or _tokenizer is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()


def generate_summary(text: str, max_length: int = 150, num_beams: int = 4) -> dict:
    """Generate a summary for a single text input.

    Returns a dict: {"task": "summarization", "model": "T5", "result": summary}
    """
    if not text:
        return {"task": "summarization", "model": "T5", "result": ""}

    # fallback simple summarization
    if not _HF_AVAILABLE:
        s = text.strip()
        first = s.split(".")[0]
        summary = first.strip()[:max_length]
        return {"task": "summarization", "model": "T5 (stub)", "result": summary}

    _load_model_if_needed()
    prefix = f"summarize: {text.strip()}"
    inputs = _tokenizer(prefix, return_tensors="pt", truncation=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = _model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    summary = _tokenizer.decode(generated[0], skip_special_tokens=True)
    return {"task": "summarization", "model": "T5", "result": summary}


class T5Handler:
    def __init__(self):
        self.model_name = "T5"

    def summarize(self, texts: List[str], max_length: int = 150, num_beams: int = 4) -> List[str]:
        results: List[str] = []
        if not texts:
            return results
        if not _HF_AVAILABLE:
            for t in texts:
                s = t.strip()
                first = s.split(".")[0]
                results.append(first.strip()[:max_length])
            return results

        _load_model_if_needed()
        for t in texts:
            prefix = f"summarize: {t.strip()}"
            inputs = _tokenizer(prefix, return_tensors="pt", truncation=True)
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            with torch.no_grad():
                generated = _model.generate(**inputs, max_length=max_length, num_beams=num_beams)
            summary = _tokenizer.decode(generated[0], skip_special_tokens=True)
            results.append(summary)
        return results

