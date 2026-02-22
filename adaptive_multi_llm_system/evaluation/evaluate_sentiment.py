"""Evaluate BERT sentiment model on a CSV file using pandas + sklearn.

This script will:
1. Load the CSV dataset at `datasets/sentiment_test.csv`
2. Predict using `analyze_sentiment` from `models.bert_handler`
3. Compare predictions with true labels and print classification report and F1

If sklearn is not available, a simple accuracy will be printed instead.
"""
import os
import sys

import pandas as pd

try:
    from sklearn.metrics import classification_report, f1_score
    _SKL_AVAILABLE = True
except Exception:
    classification_report = None
    f1_score = None
    _SKL_AVAILABLE = False

from models.bert_handler import analyze_sentiment


def evaluate(path: str = os.path.join("datasets", "sentiment_test.csv")):
    if not os.path.exists(path):
        print(f"Dataset not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        print("CSV must contain 'text' and 'label' columns")
        sys.exit(1)

    texts = df["text"].astype(str).tolist()
    true = df["label"].astype(str).str.lower().tolist()

    preds = []
    for t in texts:
        out = analyze_sentiment(t)
        # analyze_sentiment returns dict {task, model, result, score?}
        if isinstance(out, dict):
            lab = str(out.get("result", "Neutral")).lower()
        else:
            lab = str(out).lower()
        # normalize labels to 'positive'/'negative'
        if lab.startswith("pos"):
            preds.append("positive")
        elif lab.startswith("neg"):
            preds.append("negative")
        else:
            # fallback: if true label contains 'posit' mark positive
            preds.append("neutral")

    print("Sentiment evaluation")
    print("n=", len(true))
    if _SKL_AVAILABLE:
        print("Classification report:\n")
        print(classification_report(true, preds, zero_division=0))
        f1 = f1_score(true, preds, average="macro", zero_division=0)
        print(f"Macro F1: {f1:.4f}")
    else:
        # simple accuracy fallback
        correct = sum(1 for a, b in zip(true, preds) if a == b)
        acc = correct / max(1, len(true))
        print(f"Accuracy (simple): {acc:.4f}")


if __name__ == "__main__":
    evaluate()
