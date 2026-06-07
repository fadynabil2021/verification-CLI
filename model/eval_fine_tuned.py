"""model/eval_fine_tuned.py — Evaluation script for the fine-tuned LoRA model.

Loads the base model + LoRA adapters from artifacts/verilog-ai-model, runs
inference on the test split, and computes classification metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from model.prompt import format_prompt
from model.inference import classify_log  # rule-based fallback


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_output(text: str, fallback_log: str) -> Dict[str, str | float]:
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
    # If JSON parsing fails, fall back to rule-based parser.
    return classify_log(fallback_log)


def evaluate(
    test_rows: List[Dict],
    model_dir: str,
    base_model_name: str = "google/flan-t5-small",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    print(f"Loading model and tokenizer on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(base_model_name)
    is_seq2seq = getattr(config, "is_encoder_decoder", False)

    if is_seq2seq:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, model_dir)
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    details: List[Dict] = []

    print(f"Running inference on {len(test_rows)} test samples...")
    for idx, row in enumerate(test_rows):
        log = row.get("log", "")
        expected_label = row.get("label", "Unknown")

        prompt = format_prompt(log)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        
        if is_seq2seq:
            gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            input_len = inputs["input_ids"].shape[1]
            gen_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        parsed = parse_output(gen_text, log)
        predicted_label = parsed.get("label", "Unknown")

        is_correct = (predicted_label == expected_label)
        if is_correct:
            correct += 1
        total += 1

        details.append({
            "id": row.get("id", f"test_{idx}"),
            "expected": expected_label,
            "predicted": predicted_label,
            "confidence": parsed.get("confidence", 0.0),
            "correct": is_correct,
            "generated_text": gen_text,
        })

    accuracy = (correct / total) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LoRA model")
    parser.add_argument("--test-dataset", default="data/splits/test.jsonl")
    parser.add_argument("--model-dir", default="artifacts/verilog-ai-model")
    parser.add_argument("--base-model", default="google/flan-t5-small")
    parser.add_argument("--out", default="reports/fine_tuned_test_report.json")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.test_dataset))
    report = evaluate(rows, args.model_dir, args.base_model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nExact-Match Accuracy on test split: {report['accuracy'] * 100:.2f}%")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
