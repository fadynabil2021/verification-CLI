#!/usr/bin/env python3
"""
fine_tune_flan_t5.py — VeriLog AI Model Fine-tuning Script
===========================================================
Colab/GPU-ready training script for VeriLog AI.

Stage A: google/flan-t5-small (CPU or Colab Free T4)
Stage B: codellama/CodeLlama-7B-Instruct-hf (Colab Pro A100, with QLoRA)

Usage in Colab:
    !git clone https://github.com/fadynabil/verilog-ai && cd verilog-ai
    !pip install -r requirements.txt
    !python -m data.generate_dataset --no-sim --all-modules
    !python -m data.augment_dataset --n-aug 16
    !python -m data.split_dataset
    !python notebooks/fine_tune_flan_t5.py --stage A
    # After training:
    !python -m model.eval --dataset data/splits/test.jsonl

Run locally (Stage A, CPU):
    python notebooks/fine_tune_flan_t5.py --stage A --max-steps 50 --dry-run

Stage B (QLoRA, requires GPU with ≥10GB VRAM):
    python notebooks/fine_tune_flan_t5.py --stage B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Prompt / target formatters
# ---------------------------------------------------------------------------

def _format_prompt(log: str) -> str:
    return (
        "You are an expert digital IC verification engineer. "
        "Classify the following simulation failure log.\n"
        "Return a JSON object with exactly three keys:\n"
        "  'label': one of [Off-by-One Error, Handshake Protocol Violation, "
        "Data Integrity Error, Assignment Semantics Change, Edge Sensitivity Flip, "
        "Reset Polarity Inversion, Enable Signal Polarity Flip, Data Width Truncation, "
        "Overflow Guard Removal, Parity Check Removal, Unknown]\n"
        "  'explanation': a 1-2 sentence root-cause explanation for a verification engineer\n"
        "  'confidence': a float between 0.0 and 1.0\n\n"
        f"Simulation Log:\n{log}\n\n"
        "JSON:"
    )


def _format_target(row: Dict) -> str:
    return json.dumps(
        {
            "label":       row.get("label", "Unknown"),
            "explanation": row.get("explanation", "No explanation available."),
            "confidence":  float(row.get("confidence", 1.0)),
        },
        ensure_ascii=False,
    )


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Stage A: flan-t5-small (Standard fine-tuning, no quantization)
# ---------------------------------------------------------------------------

def train_stage_a(
    train_path: Path,
    val_path:   Optional[Path],
    output_dir: Path,
    model_name: str  = "google/flan-t5-small",
    max_steps:  int  = 200,
    batch_size: int  = 4,
    lr:         float = 2e-4,
    max_input:  int  = 1024,
    max_target: int  = 256,
    use_lora:   bool = False,
    dry_run:    bool = False,
) -> Dict:
    """Fine-tune a seq2seq model (flan-t5-small by default)."""
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    print(f"[Stage A] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Load data
    train_rows = _load_jsonl(train_path)
    print(f"[Stage A] Training samples: {len(train_rows)}")

    def _make_ds(rows: List[Dict]) -> Dataset:
        return Dataset.from_list([
            {"prompt": _format_prompt(r.get("log", "")),
             "target": _format_target(r)}
            for r in rows
        ])

    def _preprocess(batch: Dict) -> Dict:
        inputs = tokenizer(
            batch["prompt"], max_length=max_input, truncation=True, padding=False,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=max_target, truncation=True, padding=False,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = _make_ds(train_rows).map(_preprocess, batched=True,
                                        remove_columns=["prompt", "target"])
    val_ds   = None
    if val_path and val_path.exists():
        val_rows = _load_jsonl(val_path)
        val_ds   = _make_ds(val_rows).map(_preprocess, batched=True,
                                          remove_columns=["prompt", "target"])
        print(f"[Stage A] Validation samples: {len(val_rows)}")

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    eval_strategy = "steps" if val_ds is not None else "no"
    callbacks     = [EarlyStoppingCallback(early_stopping_patience=3)] if val_ds else []

    training_args = TrainingArguments(
        output_dir                  = str(output_dir),
        learning_rate               = lr,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = 1,
        max_steps                   = 2 if dry_run else max_steps,
        warmup_ratio                = 0.1,
        logging_steps               = 5,
        save_steps                  = 50,
        eval_steps                  = 50 if val_ds else None,
        evaluation_strategy         = eval_strategy,
        load_best_model_at_end      = val_ds is not None,
        metric_for_best_model       = "eval_loss" if val_ds else None,
        report_to                   = "none",
        fp16                        = False,  # Set True on GPU
        predict_with_generate       = False,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        tokenizer     = tokenizer,
        data_collator = collator,
        callbacks     = callbacks,
    )

    print(f"[Stage A] Starting training (max_steps={training_args.max_steps})")
    train_result = trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = train_result.metrics
    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[Stage A] Training complete. Metrics: {metrics}")
    print(f"[Stage A] Model saved to {output_dir}")

    return metrics


# ---------------------------------------------------------------------------
# Stage B: CodeLlama-7B + QLoRA (4-bit quantization, requires GPU)
# ---------------------------------------------------------------------------

def train_stage_b(
    train_path: Path,
    val_path:   Optional[Path],
    output_dir: Path,
    model_name: str  = "codellama/CodeLlama-7B-Instruct-hf",
    max_steps:  int  = 500,
    batch_size: int  = 4,
    lr:         float = 2e-4,
    max_input:  int  = 4096,
    dry_run:    bool = False,
) -> Dict:
    """QLoRA fine-tune CodeLlama-7B for causal LM classification+explanation."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, TaskType, get_peft_model
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"[Stage B] Missing dependency: {e}")
        print("[Stage B] Install: pip install transformers peft bitsandbytes trl")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[Stage B] WARNING: No CUDA GPU detected. Stage B requires GPU with ≥10GB VRAM.")
        print("[Stage B] For CPU execution use --stage A instead.")
        if not dry_run:
            sys.exit(1)

    print(f"[Stage B] Loading model (4-bit QLoRA): {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not dry_run:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_config,
            device_map          = "auto",
        )
        model.config.use_cache = False

        lora_cfg = LoraConfig(
            r             = 16,
            lora_alpha    = 32,
            lora_dropout  = 0.05,
            bias          = "none",
            task_type     = TaskType.CAUSAL_LM,
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        print("[Stage B] Dry run — skipping model load")
        model = None

    # Format as instruction-following dataset
    train_rows = _load_jsonl(train_path)
    print(f"[Stage B] Training samples: {len(train_rows)}")

    def _format_causal(row: Dict) -> str:
        return (
            f"<s>[INST] {_format_prompt(row.get('log', ''))} [/INST]\n"
            f"{_format_target(row)}</s>"
        )

    formatted_texts = [_format_causal(r) for r in train_rows]

    from datasets import Dataset as HFDataset
    train_ds = HFDataset.from_dict({"text": formatted_texts})

    output_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        sft_config = SFTConfig(
            output_dir                  = str(output_dir),
            max_seq_length              = max_input,
            num_train_epochs            = 3,
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = 4,
            learning_rate               = lr,
            warmup_ratio                = 0.1,
            lr_scheduler_type           = "cosine",
            logging_steps               = 10,
            save_steps                  = 100,
            fp16                        = False,
            bf16                        = True,
            report_to                   = "none",
            max_steps                   = max_steps,
            dataset_text_field          = "text",
        )

        trainer = SFTTrainer(
            model     = model,
            tokenizer = tokenizer,
            args      = sft_config,
            train_dataset = train_ds,
        )

        print(f"[Stage B] Starting QLoRA training (max_steps={sft_config.max_steps})")
        train_result = trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        metrics = train_result.metrics
        (output_dir / "train_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(f"[Stage B] Training complete. Metrics: {metrics}")
        return metrics

    print("[Stage B] Dry run complete — no model was trained")
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VeriLog AI fine-tuning script (Stage A: flan-t5, Stage B: CodeLlama QLoRA)"
    )
    parser.add_argument("--stage",      choices=["A", "B"], default="A",
                        help="Training stage: A=flan-t5-small, B=CodeLlama-7B QLoRA")
    parser.add_argument("--train",      default="data/splits/train.jsonl",
                        help="Training JSONL (from data/split_dataset.py)")
    parser.add_argument("--val",        default="data/splits/val.jsonl",
                        help="Validation JSONL (optional)")
    parser.add_argument("--output-dir", default="artifacts/verilog-ai-model",
                        help="Directory for checkpoints and final model")
    parser.add_argument("--max-steps",  type=int, default=200,
                        help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--use-lora",   action="store_true",
                        help="(Stage A only) Enable LoRA adapters")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Run 2 steps to validate pipeline without full training")
    args = parser.parse_args()

    train_path  = Path(args.train)
    val_path    = Path(args.val)
    output_dir  = Path(args.output_dir)

    if not train_path.exists():
        print(f"[ERROR] Training JSONL not found: {train_path}")
        print("Run first: python -m data.augment_dataset && python -m data.split_dataset")
        sys.exit(1)

    if args.stage == "A":
        train_stage_a(
            train_path = train_path,
            val_path   = val_path if val_path.exists() else None,
            output_dir = output_dir,
            max_steps  = args.max_steps,
            batch_size = args.batch_size,
            lr         = args.lr,
            use_lora   = args.use_lora,
            dry_run    = args.dry_run,
        )
    else:
        train_stage_b(
            train_path = train_path,
            val_path   = val_path if val_path.exists() else None,
            output_dir = output_dir,
            max_steps  = args.max_steps,
            batch_size = args.batch_size,
            lr         = args.lr,
            dry_run    = args.dry_run,
        )


if __name__ == "__main__":
    main()
