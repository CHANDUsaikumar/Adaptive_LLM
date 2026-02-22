"""BERT sentiment integration using Hugging Face.

Provides:
- analyze_sentiment(text: str) -> dict: runs distilbert-base-uncased-finetuned-sst-2-english
- BertHandler class with predict(texts: List[str]) -> List[dict] for backward compatibility

If the transformers/torch libraries are not available, a lightweight rule-based
fallback will be used so the project remains runnable.
"""
from typing import List, Dict
import re

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.nn.functional import softmax
    _HF_AVAILABLE = True
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    softmax = None
    _HF_AVAILABLE = False

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Global (module-level) model & tokenizer so they load once
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
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()


def analyze_sentiment(text: str) -> Dict:
    """Analyze sentiment for a single text string.

    Returns a dict: {"task": "sentiment", "model": "BERT", "result": label}
    where label is one of: Positive, Negative, or Neutral (fallback).
    """
    if not text:
        return {"task": "sentiment", "model": "BERT", "result": "Neutral"}

    # simple fallback if HF not available
    if not _HF_AVAILABLE:
        lower = text.lower()
        if re.search(r"\b(good|great|excellent|love|loved|fantastic|happy)\b", lower):
            return {"task": "sentiment", "model": "BERT (stub)", "result": "Positive"}
        if re.search(r"\b(bad|terrible|hate|hated|awful|worst|sad)\b", lower):
            return {"task": "sentiment", "model": "BERT (stub)", "result": "Negative"}
        return {"task": "sentiment", "model": "BERT (stub)", "result": "Neutral"}

    # real model path
    _load_model_if_needed()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        probs = softmax(logits, dim=1).cpu().squeeze().tolist()
        # model config maps ids to labels (e.g., 0 -> NEGATIVE, 1 -> POSITIVE)
        pred_id = int(torch.argmax(logits, dim=1).cpu().item())
        label = None
        if hasattr(_model, "config") and getattr(_model.config, "id2label", None):
            label = _model.config.id2label.get(pred_id)
        if not label:
            label = "Positive" if pred_id == 1 else "Negative"
        # Normalize label
        label = str(label).capitalize()
        return {"task": "sentiment", "model": "BERT", "result": label, "score": probs[pred_id]}


class BertHandler:
    """Compatibility wrapper used by the router and demos.

    predict(texts: List[str]) -> List[Dict] returns dicts of the form
    {"label": <label>, "score": <confidence>}
    """

    def __init__(self):
        self.model_name = "BERT"

    def predict(self, texts: List[str]) -> List[Dict]:
        results = []
        if not _HF_AVAILABLE:
            for t in texts:
                r = analyze_sentiment(t)
                lab = r.get("result", "Neutral")
                score = 0.95 if lab in ("Positive", "Negative") else 0.6
                results.append({"label": lab.lower(), "score": score})
            return results

        _load_model_if_needed()
        # Batch tokenize
        inputs = _tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = _model(**inputs)
            logits = outputs.logits
            probs = softmax(logits, dim=1).cpu().tolist()
            preds = logits.argmax(dim=1).cpu().tolist()

        for pred_id, prob in zip(preds, probs):
            if hasattr(_model, "config") and getattr(_model.config, "id2label", None):
                label = _model.config.id2label.get(pred_id, "POSITIVE" if pred_id == 1 else "NEGATIVE")
            else:
                label = "POSITIVE" if pred_id == 1 else "NEGATIVE"
            results.append({"label": label.lower(), "score": prob[pred_id]})
        return results

