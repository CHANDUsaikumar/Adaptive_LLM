"""Utility metrics used by evaluation scripts."""
from typing import List, Dict

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
except Exception:
    accuracy_score = f1_score = precision_score = recall_score = None


def simple_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if accuracy_score:
        return accuracy_score(y_true, y_pred)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / max(1, len(y_true))


def simple_f1(y_true: List[str], y_pred: List[str]) -> float:
    if f1_score:
        return f1_score(y_true, y_pred, average="macro")
    # fallback: naive f1 for binary/label sets: return 0.0 if not available
    return 0.0


def rouge_l_score(reference: str, hypothesis: str) -> float:
    # Very small and fast approximation: longest common subsequence ratio
    # This is not a full ROUGE-L implementation but serves as a quick proxy.
    r, h = reference.split(), hypothesis.split()
    if not r or not h:
        return 0.0
    # LCS dynamic programming
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if r[i] == h[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[0][0]
    return lcs / m
