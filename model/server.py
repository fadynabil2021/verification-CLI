import json
import os
from typing import Dict

from fastapi import FastAPI
from transformers import pipeline

from model.inference import classify_log

MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-small")

app = FastAPI(title="VeriLog AI Model Server")
_pipe = pipeline("text2text-generation", model=MODEL_NAME)


def _prompt(log: str) -> str:
    return (
        "You are a verification log classifier. "
        "Return JSON with keys: label, explanation, confidence (0-1). "
        "Log:\n"
        f"{log}\n"
        "JSON:"
    )


def _parse(text: str, log: str) -> Dict[str, str | float]:
    try:
        data = json.loads(text)
        if {"label", "explanation", "confidence"} <= set(data.keys()):
            return data
    except Exception:
        pass
    return classify_log(log)


@app.post("/classify")
def classify(payload: dict) -> dict:
    log = payload.get("log", "")
    output = _pipe(_prompt(log), max_new_tokens=128, do_sample=False)
    text = output[0]["generated_text"]
    return _parse(text, log)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
