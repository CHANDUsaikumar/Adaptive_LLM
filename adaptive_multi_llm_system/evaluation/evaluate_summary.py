"""Evaluate the T5 summarization model using ROUGE score.

Steps:
1. Read paragraphs from datasets/summarization_articles.txt
2. Read reference summaries from datasets/summarization_reference.txt
3. For each article generate summary using generate_summary()
4. Compute ROUGE-1 and ROUGE-L scores using rouge_score library
5. Average scores across all samples
6. Print final ROUGE scores clearly
"""

import os
import sys
from typing import List

from models.t5_handler import generate_summary

try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except Exception:
    _ROUGE_AVAILABLE = False

from evaluation.metrics_utils import rouge_l_score


def load_paragraphs(path: str) -> List[str]:
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    # Split on blank lines to get paragraphs
    parts = [p.strip() for p in content.split("\n\n") if p.strip()]
    return parts


def evaluate(articles_path: str = os.path.join("datasets", "summarization_articles.txt"),
             refs_path: str = os.path.join("datasets", "summarization_reference.txt")):
    articles = load_paragraphs(articles_path)
    refs = [r.strip() for r in open(refs_path, "r", encoding="utf-8").read().strip().splitlines() if r.strip()]

    if len(articles) != len(refs):
        print(f"Number of articles ({len(articles)}) and references ({len(refs)}) must match.")
        sys.exit(1)

    hyps = []
    for a in articles:
        out = generate_summary(a)
        if isinstance(out, dict):
            hyps.append(str(out.get("result", "")).strip())
        else:
            hyps.append(str(out).strip())

    rouge1_scores = []
    rougel_scores = []

    if _ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        for ref, hyp in zip(refs, hyps):
            scores = scorer.score(ref, hyp)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rougel_scores.append(scores["rougeL"].fmeasure)
    else:
        # Fallback: use our simple rouge_l proxy and a token overlap for rouge-1
        for ref, hyp in zip(refs, hyps):
            # ROUGE-1 proxy: unigram overlap F1
            r_tokens = ref.split()
            h_tokens = hyp.split()
            if not r_tokens or not h_tokens:
                rouge1_scores.append(0.0)
            else:
                common = 0
                ref_counts = {}
                for t in r_tokens:
                    ref_counts[t] = ref_counts.get(t, 0) + 1
                for t in h_tokens:
                    if ref_counts.get(t, 0) > 0:
                        common += 1
                        ref_counts[t] -= 1
                prec = common / max(1, len(h_tokens))
                rec = common / max(1, len(r_tokens))
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                rouge1_scores.append(f1)
            # ROUGE-L proxy from metrics_utils
            rougel_scores.append(rouge_l_score(ref, hyp))

    avg_rouge1 = sum(rouge1_scores) / max(1, len(rouge1_scores))
    avg_rougeL = sum(rougel_scores) / max(1, len(rougel_scores))

    print("Summarization evaluation")
    print(f"n = {len(hyps)}")
    print(f"Average ROUGE-1 (F1): {avg_rouge1:.4f}")
    print(f"Average ROUGE-L (proxy F1): {avg_rougeL:.4f}")


if __name__ == "__main__":
    evaluate()
