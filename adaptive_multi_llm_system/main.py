"""Entry point for the adaptive multi-LLM system demo.
Run: python -m adaptive_multi_llm_system.main
This script provides a tiny CLI that classifies prompts and routes them to the
appropriate handler (sentiment, summarization, generation).
"""
from typing import Optional

from adaptive_multi_llm_system.routing.router import classify_task, get_handler
from adaptive_multi_llm_system.utils.logger import get_logger

LOG = get_logger("main")

TASK_ALIASES = {
    "sentiment": "sentiment",
    "sentiment analysis": "sentiment",
    "sentiment-analysis": "sentiment",
    "summary": "summarization",
    "summarization": "summarization",
    "summarize": "summarization",
    "generation": "generation",
    "text generation": "generation",
    "generate": "generation",
}


def normalize_task(task: Optional[str]) -> Optional[str]:
    if not task:
        return None
    return TASK_ALIASES.get(task.strip().lower())


def process_text(task: str, text: str) -> None:
    t = normalize_task(task)
    if not t:
        LOG.error("Unsupported task '%s'", task)
        return

    handler = get_handler(t)
    model_name = getattr(handler, "model_name", handler.__class__.__name__)
    print(f"Selected model: {model_name} (handler: {handler.__class__.__name__})")

    if t == "sentiment":
        preds = handler.predict([text])
        pred = preds[0] if preds else {}
        print(f"Input: {text}\nPrediction: {pred.get('label')} (score={pred.get('score')})")
    elif t == "summarization":
        summary = handler.summarize([text])[0]
        print("Summary:\n", summary)
    else:
        gen = handler.generate([text])[0] if hasattr(handler, "generate") else ""
        print("Generation:\n", gen)


def run_cli():
    print("Adaptive Multi-LLM System CLI — type 'exit' to quit")
    while True:
        try:
            text = input("\nEnter your prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return

        task = classify_task(text)
        if task == "sentiment":
            out = get_handler("sentiment").predict([text])[0]
            print(out)
        elif task == "summarization":
            out = get_handler("summarization").summarize([text])[0]
            print(out)
        else:
            out = get_handler("generation").generate([text])[0]
            print(out)


if __name__ == "__main__":
    run_cli()
