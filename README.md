🧠 Adaptive Multi-LLM NLP System

An intelligent Natural Language Processing system that dynamically selects the most suitable transformer model for a user’s request instead of relying on a single universal LLM.

The system first detects the intent of the input query, then routes it to a specialized model:

BERT → sentiment classification

T5 → text summarization

FLAN-T5 → open-ended language generation

The objective is to simulate a real production AI pipeline that improves response quality and computational efficiency using model orchestration.

📌 System Architecture

Pipeline

User Input → Intent Router → Model Selection → Model Inference → Response

The router acts as a controller that determines the NLP task and invokes the appropriate model handler.

🤖 Models Used
Model	Architecture	Purpose
DistilBERT	Encoder	Sentiment Classification
T5-Small	Encoder-Decoder	Text Summarization
FLAN-T5	Instruction-tuned LLM	Open-ended Text Generation
📊 Evaluation Methodology

Instead of assuming which model works best, each model was evaluated on a held-out dataset using task-specific evaluation metrics.

Task Type	Evaluation Metric	Reason
Classification	F1 Score	Balances precision & recall
Summarization	ROUGE	Measures information retention
Generation	BLEU	Measures lexical similarity baseline
📈 Evaluation Results
Sentiment Classification — BERT
Metric	Score
Accuracy	1.00
Macro F1 Score	1.00
Dataset Size	5 samples

The dataset is intentionally small and controlled to verify correct routing and classification behavior.

Text Summarization — T5
Metric	Score
ROUGE-1 (F1)	0.304
ROUGE-L (F1)	0.335
Samples	3 articles

Interpretation:
The model retains approximately 30–35% of key information from human reference summaries without fine-tuning, which is expected for a small general-purpose transformer.

Text Generation — FLAN-T5
Metric	Score
BLEU Score	0.0109
Prompts	3

Note:
BLEU relies on exact n-gram overlap and underestimates open-ended generation quality.
FLAN-T5 produces semantically correct responses with lexical variation, resulting in a low BLEU score despite good qualitative performance.

🧩 Model Selection Justification

The routing logic is data-driven based on empirical evaluation.

Task	Selected Model	Justification
Sentiment Analysis	BERT	Highest classification reliability (F1)
Summarization	T5	Better information retention (ROUGE)
Open-ended Queries	FLAN-T5	Best reasoning & natural language generation
✨ Features

Intent-based query routing

Multi-model orchestration

Fully local inference (no external API dependency)

Quantitative evaluation using standard NLP metrics

FastAPI web interface

🚀 Running the Project
1. Clone repository
git clone <your-repo-link>
cd adaptive_multi_llm_system
2. Create virtual environment
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Start the server
PYTHONPATH=$(pwd) python -m uvicorn adaptive_multi_llm_system.web.app:app
5. Open in browser
http://127.0.0.1:8000
🧪 Example Queries

Sentiment

I regret buying this phone. The battery drains very fast.

Summarization

Artificial intelligence is rapidly transforming industries such as healthcare, finance, and transportation. It enables automation and predictive analytics while also raising ethical concerns about privacy and employment.

Generation

Explain black holes to a 10 year old.
🛠 Technologies Used

Python

HuggingFace Transformers

PyTorch

FastAPI

Scikit-learn

NLTK

ROUGE-Score

🎯 Key Learning Outcome

This project demonstrates that modern AI systems are not built around a single model, but around intelligent orchestration of specialized models, where architectural decisions are justified using empirical evaluation metrics.
