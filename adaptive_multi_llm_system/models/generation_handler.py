"""Compatibility wrapper for generation models.
Delegates to the existing FLAN-T5 handler implementation.
"""
from typing import List

try:
    from adaptive_multi_llm_system.models.gpt_flant5_handler import generate_text as _gen_text, FlanT5Handler
except Exception:
    # fall back to relative import if package layout is different
    try:
        from models.gpt_flant5_handler import generate_text as _gen_text, FlanT5Handler
    except Exception:
        _gen_text = None
        FlanT5Handler = None


def generate_text(prompt: str, **kwargs):
    if _gen_text is None:
        # simple fallback
        return {"task": "generation", "model": "stub", "result": (prompt or "") + " (simulated)"}
    return _gen_text(prompt, **kwargs)


def generate(prompts: List[str], **kwargs) -> List[str]:
    if FlanT5Handler is None:
        return [(p or "") + " (simulated)" for p in prompts]
    handler = FlanT5Handler()
    return handler.generate(prompts, **kwargs)
