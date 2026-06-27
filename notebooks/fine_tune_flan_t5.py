#!/usr/bin/env python3
"""
fine_tune_flan_t5.py — VeriLog AI: Stage A Fine-Tuning Guide (Colab-Ready)
==========================================================================

End-to-end fine-tuning script for google/flan-t5-small on the VeriLog AI
verification log classification dataset.

┌─────────────────────────────────────────────────────────────────────────┐
│  QUICK START (Google Colab — paste into the first cell)                │
│                                                                        │
│  # Cell 1: Clone & install                                             │
│  !git clone https://github.com/fadynabil/verilog-ai.git                │
│  %cd verilog-ai                                                        │
│  !pip install -q -U transformers datasets peft accelerate torch        │
│                                                                        │
│  # Cell 2: Generate data pipeline (no Verilator required)              │
│  !PYTHONPATH=. python -m data.generate_dataset --no-sim --all-modules  │
│  !PYTHONPATH=. python -m data.augment_dataset --n-aug 16               │
│  !PYTHONPATH=. python -m data.split_dataset                            │
│                                                                        │
│  # Cell 3: Fine-tune (≈5 min on T4 GPU, ≈20 min on CPU)               │
│  !PYTHONPATH=. python notebooks/fine_tune_flan_t5.py                   │
│                                                                        │
│  # Cell 4: Evaluate on held-out test set                               │
│  !PYTHONPATH=. python -m model.eval_fine_tuned                         │
│                                                                        │
│  # Cell 5 (optional): Save model to Google Drive                       │
│  !PYTHONPATH=. python notebooks/fine_tune_flan_t5.py --save-to-drive   │
└─────────────────────────────────────────────────────────────────────────┘

Architecture:
    This script trains a Seq2Seq model (Encoder-Decoder) that takes a
    simulation log as input and generates a JSON string with three keys:
        {label, explanation, confidence}

    The prompt template is loaded from model/prompt.py (shared with the
    model server and CLI) to prevent train/serve skew.

Bugs fixed from v1:
    - FIXED: Prompt skew — now imports shared format_prompt from model.prompt
    - FIXED: deprecated `evaluation_strategy` → `eval_strategy`
    - FIXED: missing `save_strategy` argument
    - FIXED: fp16 hardcoded to False — now auto-detects GPU
    - FIXED: no Google Drive checkpointing for Colab disconnect resilience
    - FIXED: EarlyStoppingCallback imported but crashes without
             `load_best_model_at_end=True` when no val set exists
    - ADDED: --save-to-drive flag for post-training Drive backup
    - ADDED: --use-lora LoRA adapter support with proper PEFT save
    - ADDED: rich progress logging with dataset & split statistics
    - ADDED: automatic Colab environment detection
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _is_colab() -> bool:
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _has_gpu() -> bool:
    """Check for CUDA GPU availability."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Shared prompt template (prevents train/serve skew)
# ---------------------------------------------------------------------------

try:
    # When running from repo root with PYTHONPATH=.
    from model.prompt import format_prompt as _format_prompt
except ImportError:
    # Fallback: inline the EXACT same prompt from model/prompt.py.
    # ⚠️  If you change model/prompt.py, you MUST update this fallback too.
    def _format_prompt(log: str) -> str:
        return (
            "You are a verification log classifier. "
            "Return JSON with keys: label, explanation, confidence (0-1). "
            "Log:\n"
            f"{log}\n"
            "JSON:"
        )
    print(
        "[WARN] Could not import model.prompt.format_prompt. "
        "Using inline fallback. Run with PYTHONPATH=. to avoid skew."
    )


def _format_target(row: Dict) -> str:
    """Format the target JSON string for supervised training."""
    return json.dumps(
        {
            "label":       row.get("label", "Unknown"),
            "explanation": row.get("explanation", "No explanation available."),
            "confidence":  float(row.get("confidence", 1.0)),
        },
        ensure_ascii=False,
    )


def _load_jsonl(path: Path) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _print_label_distribution(rows: List[Dict], name: str) -> None:
    """Print label distribution for a dataset split."""
    from collections import Counter
    labels = Counter(r.get("label", "Unknown") for r in rows)
    print(f"\n  [{name}] {len(rows)} samples — label distribution:")
    for label, count in sorted(labels.items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 2, 40)
        print(f"    {label:<32} {count:>4}  {bar}")


# ---------------------------------------------------------------------------
# Google Drive integration (Colab disconnect resilience)
# ---------------------------------------------------------------------------

def _save_to_drive(source_dir: Path, drive_subdir: str = "verilog-ai-model") -> None:
    """Copy model artifacts to Google Drive for persistence across sessions."""
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        dest = Path("/content/drive/MyDrive") / drive_subdir
        dest.mkdir(parents=True, exist_ok=True)
        # Copy all files from source to destination
        for f in source_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)
            elif f.is_dir():
                dst_sub = dest / f.name
                if dst_sub.exists():
                    shutil.rmtree(dst_sub)
                shutil.copytree(f, dst_sub)
        print(f"[DRIVE] Model saved to Google Drive: {dest}")
    except ImportError:
        print("[DRIVE] Not in Colab — skipping Drive save.")
    except Exception as e:
        print(f"[DRIVE] Failed to save to Drive: {e}")


# ---------------------------------------------------------------------------
# Stage A: flan-t5-small fine-tuning
# ---------------------------------------------------------------------------

def train_stage_a(
    train_path: Path,
    val_path:   Optional[Path],
    output_dir: Path,
    model_name: str   = "google/flan-t5-small",
    max_steps:  int   = 200,
    batch_size: int   = 4,
    lr:         float = 2e-4,
    max_input:  int   = 1024,
    max_target: int   = 256,
    use_lora:   bool  = False,
    dry_run:    bool  = False,
    save_drive: bool  = False,
) -> Dict:
    """Fine-tune a Seq2Seq model (flan-t5-small) for log classification.

    Returns training metrics dict.
    """
    # Lazy imports — keeps startup fast when just running --help
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    gpu_available = _has_gpu()
    print("=" * 72)
    print(f"  VeriLog AI — Stage A Fine-Tuning")
    print(f"  Model:     {model_name}")
    print(f"  Device:    {'GPU (' + __import__('torch').cuda.get_device_name(0) + ')' if gpu_available else 'CPU'}")
    print(f"  LoRA:      {'Enabled' if use_lora else 'Disabled'}")
    print(f"  Max Steps: {2 if dry_run else max_steps} {'(dry run)' if dry_run else ''}")
    print(f"  Batch:     {batch_size}")
    print(f"  LR:        {lr}")
    print("=" * 72)

    # ── 1. Load tokenizer & model ──────────────────────────────────────
    print("\n[1/5] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_cfg)
        print("  LoRA adapters attached:")
        model.print_trainable_parameters()

    # Move to GPU if available (non-LoRA full fine-tune)
    if gpu_available and not use_lora:
        model = model.to("cuda")

    # ── 2. Load and preprocess data ────────────────────────────────────
    print("\n[2/5] Loading training data...")
    train_rows = _load_jsonl(train_path)
    if not train_rows:
        print(f"[ERROR] No rows in {train_path}. Run the data pipeline first:")
        print("  PYTHONPATH=. python -m data.generate_dataset --no-sim --all-modules")
        print("  PYTHONPATH=. python -m data.augment_dataset --n-aug 16")
        print("  PYTHONPATH=. python -m data.split_dataset")
        sys.exit(1)
    _print_label_distribution(train_rows, "Train")

    def _make_ds(rows: List[Dict]) -> Dataset:
        return Dataset.from_list([
            {"prompt": _format_prompt(r.get("log", "")),
             "target": _format_target(r)}
            for r in rows
        ])

    def _preprocess(batch: Dict) -> Dict:
        inputs = tokenizer(
            batch["prompt"],
            max_length=max_input,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=max_target,
            truncation=True,
            padding=False,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = _make_ds(train_rows).map(
        _preprocess, batched=True, remove_columns=["prompt", "target"]
    )

    val_ds = None
    if val_path and val_path.exists():
        val_rows = _load_jsonl(val_path)
        if val_rows:
            val_ds = _make_ds(val_rows).map(
                _preprocess, batched=True, remove_columns=["prompt", "target"]
            )
            _print_label_distribution(val_rows, "Val")

    # ── 3. Configure training ──────────────────────────────────────────
    print("\n[3/5] Configuring trainer...")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True,
    )

    has_val = val_ds is not None
    callbacks = []
    if has_val:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    training_args = TrainingArguments(
        output_dir                  = str(output_dir),
        learning_rate               = lr,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = 1,
        max_steps                   = 2 if dry_run else max_steps,
        warmup_ratio                = 0.1,
        logging_steps               = 5,
        # Save strategy
        save_strategy               = "steps",
        save_steps                  = 50,
        save_total_limit            = 2,
        # Eval strategy (note: `evaluation_strategy` is deprecated)
        eval_strategy               = "steps" if has_val else "no",
        eval_steps                  = 50 if has_val else None,
        # Best model selection
        load_best_model_at_end      = has_val,
        metric_for_best_model       = "eval_loss" if has_val else None,
        greater_is_better           = False if has_val else None,
        # Performance
        fp16                        = gpu_available,
        # Reporting
        report_to                   = "none",
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

    # ── 4. Train ───────────────────────────────────────────────────────
    print(f"\n[4/5] Training started (max_steps={training_args.max_steps})...")
    train_result = trainer.train()

    # ── 5. Save model & metrics ────────────────────────────────────────
    print("\n[5/5] Saving model and metrics...")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = train_result.metrics
    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print(f"  ✅ Training complete!")
    print(f"  Model saved to:   {output_dir}")
    print(f"  Metrics saved to: {metrics_path}")
    print(f"  Train loss:       {metrics.get('train_loss', 'N/A')}")
    print(f"{'=' * 72}")

    # Optional: save to Google Drive
    if save_drive:
        _save_to_drive(output_dir, "verilog-ai-flan-t5-model")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VeriLog AI — Stage A fine-tuning (flan-t5-small)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training on Colab (GPU):
  PYTHONPATH=. python notebooks/fine_tune_flan_t5.py

  # Quick validation (2 steps, CPU ok):
  PYTHONPATH=. python notebooks/fine_tune_flan_t5.py --dry-run

  # With LoRA adapters:
  PYTHONPATH=. python notebooks/fine_tune_flan_t5.py --use-lora

  # Custom dataset + save to Drive:
  PYTHONPATH=. python notebooks/fine_tune_flan_t5.py \\
      --train data/splits/train.jsonl \\
      --val data/splits/val.jsonl \\
      --save-to-drive
""",
    )
    parser.add_argument("--train",      default="data/splits/train.jsonl",
                        help="Training JSONL path (default: data/splits/train.jsonl)")
    parser.add_argument("--val",        default="data/splits/val.jsonl",
                        help="Validation JSONL path (default: data/splits/val.jsonl)")
    parser.add_argument("--output-dir", default="artifacts/verilog-ai-model",
                        help="Directory for checkpoints and final model")
    parser.add_argument("--model-name", default="google/flan-t5-small",
                        help="HuggingFace model ID (default: google/flan-t5-small)")
    parser.add_argument("--max-steps",  type=int, default=200,
                        help="Maximum training steps (default: 200)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size (default: 4)")
    parser.add_argument("--lr",         type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--use-lora",   action="store_true",
                        help="Enable LoRA adapters (smaller checkpoint, fewer trainable params)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Run 2 steps to validate the pipeline without full training")
    parser.add_argument("--save-to-drive", action="store_true",
                        help="Copy trained model to Google Drive after training")
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path   = Path(args.val)
    output_dir = Path(args.output_dir)

    if not train_path.exists():
        print(f"[ERROR] Training data not found: {train_path}")
        print("\nRun the data pipeline first:")
        print("  PYTHONPATH=. python -m data.generate_dataset --no-sim --all-modules")
        print("  PYTHONPATH=. python -m data.augment_dataset --n-aug 16")
        print("  PYTHONPATH=. python -m data.split_dataset")
        sys.exit(1)

    train_stage_a(
        train_path = train_path,
        val_path   = val_path if val_path.exists() else None,
        output_dir = output_dir,
        model_name = args.model_name,
        max_steps  = args.max_steps,
        batch_size = args.batch_size,
        lr         = args.lr,
        use_lora   = args.use_lora,
        dry_run    = args.dry_run,
        save_drive = args.save_to_drive,
    )


if __name__ == "__main__":
    main()
