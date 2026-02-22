# adaptive-multi-llm-system

This repository is a small scaffold for an adaptive multi-LLM routing and evaluation system. It provides:

- Lightweight handler stubs for different model types in `models/`:
  - `bert_handler.py` (sentiment)
  - `t5_handler.py` (summarization)
  - `gpt_handler.py` (generation)
- A router in `routing/router.py` to select handlers by task
- Evaluation scripts in `evaluation/` that use simple metrics/proxies
- Small placeholder datasets in `datasets/`
- Utility modules in `utils/`
- `main.py` to run quick demos or evaluations

How to run

1. Create a virtual environment and install requirements (optional for stubs):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run a demo (no heavy models needed):

```bash
python main.py --task sentiment --mode demo
python main.py --task summarization --mode demo
python main.py --task generation --mode demo
```

3. Run evaluations (uses stub logic and placeholder datasets):

```bash
python main.py --task sentiment --mode evaluate
python main.py --task summarization --mode evaluate
python main.py --task generation --mode evaluate
```

Next steps

- Replace stubs in `models/` with real model-loading code (Hugging Face, local LLMs, or API wrappers).
- Improve evaluation metrics (use `rouge`, `sacrebleu`, or human evals).
- Add a routing policy that picks models based on input size, cost, or accuracy.
