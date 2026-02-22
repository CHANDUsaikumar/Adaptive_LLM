"""FLAN-T5 generation handler using Hugging Face Transformers.

Provides:
- generate_text(prompt: str, temperature=1.0, max_length=128, num_beams=4)
  -> dict {"task":"generation","model":"FLAN-T5","result": generated_text}
- FlanT5Handler.generate(prompts: List[str]) -> List[str]

If transformers/torch are not installed, a lightweight stub generation is used.
"""
from typing import List

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _HF_AVAILABLE = True
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    _HF_AVAILABLE = False

MODEL_NAME = "google/flan-t5-base"

# module-level resources
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


def generate_text(prompt: str, temperature: float = 1.0, max_length: int = 128, num_beams: int = 4) -> dict:
    """Generate text for a single prompt using FLAN-T5.

    Returns: {"task":"generation","model":"FLAN-T5","result": generated_text}
    """
    if not prompt:
        return {"task": "generation", "model": "FLAN-T5", "result": ""}

    if not _HF_AVAILABLE:
        # fallback: echo with a canned extension
        gen = (prompt.strip() + "\n\nThis is a simulated FLAN-T5 generation.")[:max_length]
        return {"task": "generation", "model": "FLAN-T5 (stub)", "result": gen}

    _load_model_if_needed()
    prefix = prompt.strip()
    inputs = _tokenizer(prefix, return_tensors="pt", truncation=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = _model.generate(**inputs, max_length=max_length, num_beams=num_beams, do_sample=(temperature>0), temperature=temperature)
    text = _tokenizer.decode(generated[0], skip_special_tokens=True)
    return {"task": "generation", "model": "FLAN-T5", "result": text}


class FlanT5Handler:
    """Compatibility wrapper for batch generation."""

    def __init__(self):
        self.model_name = "FLAN-T5"

    def generate(self, prompts: List[str], temperature: float = 1.0, max_length: int = 128, num_beams: int = 4) -> List[str]:
        results: List[str] = []
        if not prompts:
            return results
        if not _HF_AVAILABLE:
            for p in prompts:
                results.append((p.strip() + " This is a simulated FLAN-T5 generation.")[:max_length])
            return results

        _load_model_if_needed()
        for p in prompts:
            prefix = p.strip()
            inputs = _tokenizer(prefix, return_tensors="pt", truncation=True)
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            with torch.no_grad():
                generated = _model.generate(**inputs, max_length=max_length, num_beams=num_beams, do_sample=(temperature>0), temperature=temperature)
            text = _tokenizer.decode(generated[0], skip_special_tokens=True)
            results.append(text)
        return results
