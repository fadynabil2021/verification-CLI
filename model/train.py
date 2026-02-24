from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _format_prompt(log: str) -> str:
    return (
        "You are a verification log classifier. "
        "Return JSON with keys: label, explanation, confidence (0-1). "
        "Log:\n"
        f"{log}\n"
        "JSON:"
    )


def _format_target(row: Dict) -> str:
    return json.dumps(
        {
            "label": row.get("label", "Unknown"),
            "explanation": row.get("explanation", "No explanation available."),
            "confidence": float(row.get("confidence", 1.0)),
        },
        ensure_ascii=False,
    )


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a seq2seq model for verification log classification"
    )
    parser.add_argument("--dataset", default="data/dataset.jsonl", help="Training JSONL")
    parser.add_argument(
        "--model-name",
        default="google/flan-t5-small",
        help="HF model ID (seq2seq model recommended)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/verilog-ai-model",
        help="Directory for checkpoints and final model",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA adapters (requires peft)",
    )
    args = parser.parse_args()

    # Import heavy deps lazily so other workflows can run without ML stack.
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    rows = _load_jsonl(Path(args.dataset))
    if not rows:
        raise ValueError(f"No rows found in dataset: {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_cfg)

    ds_rows = [{"prompt": _format_prompt(r.get("log", "")), "target": _format_target(r)} for r in rows]
    dataset = Dataset.from_list(ds_rows)

    def preprocess(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        inputs = tokenizer(
            batch["prompt"],
            max_length=args.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=args.max_target_length,
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["prompt", "target"])
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    train_output = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = Path(args.output_dir) / "train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(train_output.metrics, indent=2), encoding="utf-8")
    print(f"Training complete. Model saved to {args.output_dir}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
