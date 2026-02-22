from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
from fastapi.responses import HTMLResponse

from adaptive_multi_llm_system.routing.router import classify_task
from adaptive_multi_llm_system.models.bert_handler import analyze_sentiment
from adaptive_multi_llm_system.models.t5_handler import generate_summary
from adaptive_multi_llm_system.models.generation_handler import generate_text

app = FastAPI()

_THIS_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_THIS_DIR / "templates"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, user_input: str = Form(...)):

    task = classify_task(user_input)

    if task == "sentiment":
        result = analyze_sentiment(user_input)
    elif task == "summarization":
        result = generate_summary(user_input)
    else:
        result = generate_text(user_input)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "input": user_input,
            "task": task,
            "model": result.get("model") if isinstance(result, dict) else None,
            "output": result.get("result") if isinstance(result, dict) else result,
        },
    )
