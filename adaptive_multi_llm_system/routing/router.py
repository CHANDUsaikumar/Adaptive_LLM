"""Simple router that selects a handler by task type and includes a tiny
intent classifier helper `classify_task`.

The classify_task function implements lightweight, rule-based intent
classification to decide whether input text requests a summarization,
sentiment analysis, or general text generation request.
"""
from typing import Optional
import re

from adaptive_multi_llm_system.models.bert_handler import BertHandler
from adaptive_multi_llm_system.models.t5_handler import T5Handler
from adaptive_multi_llm_system.models.generation_handler import FlanT5Handler as GPTHandler

SUPPORTED_TASKS = ["sentiment", "summarization", "generation"]


_SUMMARIZATION_PATTERNS = re.compile(
    r"\b(summarize|summarise|summary|shorten|short|brief|condense|tl;dr|explain)\b",
    re.IGNORECASE,
)

_SENTIMENT_PATTERNS = re.compile(
    r"\b(opinion|review|rate|rating|sentiment|positive|negative|like|liked|love|loved|dislike|disliked|hate|hated|good|bad|great|terrible|awful)\b",
    re.IGNORECASE,
)


def classify_task(text: str) -> str:
    """Classify user intent into one of: 'sentiment', 'summarization', 'generation'.

    Rules (case-insensitive, robust to extra spaces):
    - If text asks to summarize, shorten, brief, or explain a long passage -> 'summarization'
    - If text gives an opinion/review or asks positive/negative sentiment -> 'sentiment'
    - Otherwise -> 'generation'

    Args:
        text: input user text
    Returns:
        one of the strings: 'sentiment', 'summarization', 'generation'
    """
    if not text:
        return "generation"

    t = " ".join(text.split())  # collapse whitespace
    # Prioritize summarization keywords
    if _SUMMARIZATION_PATTERNS.search(t):
        return "summarization"

    # Then sentiment clues
    if _SENTIMENT_PATTERNS.search(t):
        return "sentiment"

    # Default
    return "generation"


def get_handler(task: str):
    """Return a handler instance for the requested task.

    Args:
        task: one of 'sentiment', 'summarization', 'generation'
    Returns:
        instance of handler class
    Raises:
        ValueError for unsupported tasks
    """
    t = task.lower()
    if t == "sentiment":
        return BertHandler()
    if t == "summarization":
        return T5Handler()
    if t == "generation":
        return GPTHandler()
    raise ValueError(f"Unsupported task: {task}. Supported: {SUPPORTED_TASKS}")
