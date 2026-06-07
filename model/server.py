import json
import os
from typing import Dict

from fastapi import FastAPI
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline,
)
from peft import PeftModel
import torch

from model.inference import classify_log
from model.prompt import format_prompt as _prompt

MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-small")

app = FastAPI(title="VeriLog AI Model Server")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Check if MODEL_NAME contains PEFT adapter config
is_peft = False
base_model_path = MODEL_NAME
if os.path.isdir(MODEL_NAME) and os.path.exists(os.path.join(MODEL_NAME, "adapter_config.json")):
    is_peft = True
    with open(os.path.join(MODEL_NAME, "adapter_config.json"), "r") as f:
        adapter_cfg = json.load(f)
        base_model_path = adapter_cfg.get("base_model_name_or_path")

# Get configuration of base/main model
config = AutoConfig.from_pretrained(base_model_path)
is_seq2seq = getattr(config, "is_encoder_decoder", False)

# Load base model
if is_seq2seq:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
else:
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    base_model.config.pad_token_id = tokenizer.pad_token_id

# Wrap with PEFT if adapter
if is_peft:
    model = PeftModel.from_pretrained(base_model, MODEL_NAME)
else:
    model = base_model

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Set up pipeline
task = "text2text-generation" if is_seq2seq else "text-generation"
_pipe = pipeline(task, model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)


def _parse(text: str, log: str) -> Dict[str, str | float]:
    try:
        text = text.strip()
        if not text.startswith("{"):
            idx = text.find("{")
            if idx != -1:
                text = text[idx:]
        if not text.endswith("}"):
            idx = text.rfind("}")
            if idx != -1:
                text = text[:idx+1]
        data = json.loads(text)
        if {"label", "explanation", "confidence"} <= set(data.keys()):
            return data
    except Exception:
        pass
    return classify_log(log)


@app.post("/classify")
def classify(payload: dict) -> dict:
    log = payload.get("log", "")
    
    # 1. Validation: Prevent OOM on model server
    MAX_INPUT_CHARS = 50000
    if len(log) > MAX_INPUT_CHARS:
        log = "... [TRUNCATED] ...\n" + log[-MAX_INPUT_CHARS:]

    prompt = _prompt(log)
    if is_seq2seq:
        output = _pipe(prompt, max_new_tokens=128, do_sample=False)
    else:
        # return_full_text=False ensures causal models return only the generated JSON
        output = _pipe(prompt, max_new_tokens=128, do_sample=False, return_full_text=False)
    
    text = output[0]["generated_text"]
    return _parse(text, log)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
