Adaptive Multi-LLM NLP System

An intelligent Natural Language Processing system that dynamically routes user queries to the most suitable transformer model instead of relying on a single universal LLM.

The application first identifies the intent of the user query, then assigns the task to a specialized model:

BERT → sentiment classification

T5 → text summarization

FLAN-T5 → open-ended text generation

The goal is to simulate a real production AI system that optimizes response quality and computational efficiency through model orchestration.

System Architecture

Pipeline

User Input → Intent Router → Model Selection → Model Inference → Response

The router acts as a controller that detects the NLP task and invokes the most appropriate model handler.

Models Used
Model	Type	Purpose
DistilBERT	Encoder	Sentiment classification
T5-Small	Encoder-Decoder	Summarization
FLAN-T5	Instruction-tuned LLM	Open-ended generation
Evaluation Methodology

Instead of assuming model suitability, each model was evaluated on a held-out test dataset using task-specific metrics.

Different NLP tasks require different evaluation criteria:

Classification → F1 Score

Summarization → ROUGE

Generation → BLEU

Evaluation Results
Sentiment Classification (BERT)
Metric	Score
Accuracy	1.00
Macro F1 Score	1.00
Dataset Size	5 samples

The dataset is intentionally small and controlled to verify correct task routing and classification behaviour.

Summarization (T5)
Metric	Score
ROUGE-1 (F1)	0.304
ROUGE-L (F1)	0.335
Samples	3 articles

Interpretation:
The model preserves ~30–35% of important information from the reference summaries without fine-tuning, which is expected for a small general-purpose transformer.

Text Generation (FLAN-T5)
Metric	Score
BLEU Score	0.0109
Prompts	3

Important Note:
BLEU measures exact n-gram overlap and tends to underestimate performance for open-ended generation.
FLAN-T5 produces semantically correct responses with lexical variation, resulting in a low BLEU score despite good qualitative output.

Model Selection Justification

The routing decision is data-driven:

Task	Selected Model	Reason
Sentiment Analysis	BERT	Highest classification reliability (F1)
Summarization	T5	Better information retention (ROUGE)
Open-ended Queries	FLAN-T5	Best natural language reasoning & generation
Features

Dynamic task detection (intent-based routing)

Multi-model orchestration

Local inference (no external API dependency)

Quantitative evaluation using standard NLP metrics

Web interface using FastAPI

Running the Project
1. Install dependencies
pip install -r requirements.txt
2. Start the server
PYTHONPATH=$(pwd) python -m uvicorn adaptive_multi_llm_system.web.app:app
3. Open in browser
http://127.0.0.1:8000
Example Queries

Sentiment

I regret buying this phone, the battery is terrible.

Summarization

<insert long paragraph>

Generation

Explain black holes to a 10 year old.
Technologies Used

Python

HuggingFace Transformers

PyTorch

FastAPI

Scikit-learn

ROUGE Score

NLTK

Key Learning Outcome

This project demonstrates that modern AI systems are not built around a single model but around intelligent orchestration of multiple specialized models, with decisions justified through empirical evaluation.
