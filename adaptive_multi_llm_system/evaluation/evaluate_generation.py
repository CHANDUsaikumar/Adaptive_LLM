"""Evaluate the FLAN-T5 generation model using BLEU score.

Steps:
1. Read prompts from datasets/generation_prompts.txt
2. Read reference responses from datasets/generation_references.txt
3. Generate model responses using the router's generation handler
4. Tokenize outputs using nltk.word_tokenize (fallback to simple split)
5. Compute BLEU score using nltk.translate.bleu_score (fallback proxy if nltk missing)
6. Average BLEU across all prompts
7. Print final BLEU score
"""

import os
import sys
from typing import List

try:
	import nltk
	from nltk.tokenize import word_tokenize
	from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
	_NLTK_AVAILABLE = True
except Exception:
	nltk = None
	word_tokenize = None
	sentence_bleu = None
	SmoothingFunction = None
	_NLTK_AVAILABLE = False

from routing.router import get_handler


def load_lines(path: str) -> List[str]:
	if not os.path.exists(path):
		print(f"Missing file: {path}")
		sys.exit(1)
	with open(path, "r", encoding="utf-8") as f:
		lines = [l.strip() for l in f.readlines() if l.strip()]
	return lines


def tokenize(text: str) -> List[str]:
	if _NLTK_AVAILABLE and word_tokenize is not None:
		try:
			# Ensure punkt is available (download if missing)
			try:
				nltk.data.find("tokenizers/punkt")
			except Exception:
				try:
					nltk.download("punkt")
				except Exception:
					pass
			return word_tokenize(text)
		except LookupError:
			# Some NLTK installs may require additional resources; fallback to simple split
			return text.split()
	# fallback
	return text.split()


def evaluate(prompts_path: str = os.path.join("datasets", "generation_prompts.txt"),
			 refs_path: str = os.path.join("datasets", "generation_references.txt")):
	prompts = load_lines(prompts_path)
	refs = load_lines(refs_path)

	if len(prompts) != len(refs):
		print(f"Number of prompts ({len(prompts)}) and references ({len(refs)}) must match.")
		sys.exit(1)

	handler = get_handler("generation")

	# Generate: prefer batch API `generate`, otherwise call single prompt generator
	gens: List[str] = []
	if hasattr(handler, "generate"):
		try:
			gens = handler.generate(prompts)
		except Exception:
			# fallback to per-prompt
			gens = [getattr(handler, "generate_text", lambda p: {"result": ""})(p).get("result", "") if hasattr(handler, "generate_text") else "" for p in prompts]
	else:
		if hasattr(handler, "generate_text"):
			gens = [handler.generate_text(p).get("result", "") for p in prompts]
		else:
			# Last resort: try calling a top-level function in models.generation_handler
			try:
				from adaptive_multi_llm_system.models.generation_handler import generate_text as _gen

				gens = [_gen(p).get("result", "") for p in prompts]
			except Exception:
				gens = ["" for _ in prompts]

	bleu_scores = []

	if _NLTK_AVAILABLE and sentence_bleu is not None:
		smoother = SmoothingFunction().method1
		for ref, hyp in zip(refs, gens):
			ref_tokens = tokenize(ref)
			hyp_tokens = tokenize(hyp)
			# sentence_bleu expects list of reference token lists
			try:
				score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1)
			except Exception:
				score = 0.0
			bleu_scores.append(score)
	else:
		# Fallback: unigram precision as a proxy for BLEU
		for ref, hyp in zip(refs, gens):
			r = tokenize(ref)
			h = tokenize(hyp)
			if not h:
				bleu_scores.append(0.0)
				continue
			ref_counts = {}
			for t in r:
				ref_counts[t] = ref_counts.get(t, 0) + 1
			match = 0
			for t in h:
				if ref_counts.get(t, 0) > 0:
					match += 1
					ref_counts[t] -= 1
			precision = match / max(1, len(h))
			bleu_scores.append(precision)

	avg_bleu = sum(bleu_scores) / max(1, len(bleu_scores))

	print("Generation evaluation")
	print(f"n = {len(gens)}")
	print(f"Average BLEU (0-1): {avg_bleu:.4f}")


if __name__ == "__main__":
	evaluate()

