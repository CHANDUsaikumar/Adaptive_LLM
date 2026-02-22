"""Basic preprocessing utilities."""
import re


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t
