#!/usr/bin/env python3
"""
fine_tune_codellama.py — VeriLog AI: Stage B QLoRA Training + Merge + Benchmark
================================================================================

Fine-tunes CodeLlama-7B-Instruct on verification log classification using QLoRA,
merges adapters into base weights, and benchmarks inference latency on T4 GPU.

Quick Start (Google Colab T4):
    # Cell 1: Clone & install
    !git clone https://github.com/fadynabil/verilog-ai.git
    %cd verilog-ai
    !pip install -q -U torch bitsandbytes transformers peft accelerate datasets trl

    # Cell 2: Data pipeline
    !PYTHONPATH=. python -m data.generate_dataset --no-sim --all-modules
    !PYTHONPATH=. python -m data.augment_dataset --n-aug 16
    !PYTHONPATH=. python -m data.split_dataset

    # Cell 3: Train + Merge (≈25 min on T4)
    !PYTHONPATH=. python notebooks/fine_tune_codellama.py --train --merge

    # Cell 4: Benchmark accuracy + latency
    !PYTHONPATH=. python notebooks/fine_tune_codellama.py --benchmark

    # Cell 5: Save to Google Drive
    !PYTHONPATH=. python notebooks/fine_tune_codellama.py --save-to-drive

Bugs fixed from v1:
    - FIXED: Prompt skew — now imports shared format_prompt from model.prompt
    - FIXED: deprecated `evaluation_strategy` → `eval_strategy`
    - FIXED: `save_steps=None` crash when save_strategy="epoch"
    - FIXED: no Google Drive checkpoint saving for Colab disconnect resilience
    - FIXED: bfloat16 compute_dtype on T4 (T4 lacks native BF16 — use float16)
    - ADDED: --save-to-drive flag for post-training Drive backup
    - ADDED: explicit memory cleanup between train/merge/benchmark phases
    - ADDED: rich progress logging with dataset statistics
    - ADDED: benchmark saves results to reports/benchmark_report.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Shared prompt template (prevents train/serve skew)
# ---------------------------------------------------------------------------
try:
    from model.prompt import format_prompt
except ImportError:
    # Fallback: MUST match model/prompt.py exactly
    def format_prompt(log: str) -> str:
        return (
            "You are a verification log classifier. "
            "Return JSON with keys: label, explanation, confidence (0-1). "
            "Log:\n"
            f"{log}\n"
            "JSON:"
        )
    print("[WARN] Could not import model.prompt — using inline fallback.")


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


def _print_label_distribution(rows: List[Dict], name: str) -> None:
    from collections import Counter
    labels = Counter(r.get("label", "Unknown") for r in rows)
    print(f"\n  [{name}] {len(rows)} samples:")
    for label, count in sorted(labels.items(), key=lambda x: -x[1]):
        print(f"    {label:<32} {count:>4}")


def _cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_to_drive(source_dir: Path, drive_subdir: str) -> None:
    """Copy model artifacts to Google Drive."""
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        dest = Path("/content/drive/MyDrive") / drive_subdir
        dest.mkdir(parents=True, exist_ok=True)
        for f in source_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)
            elif f.is_dir():
                dst_sub = dest / f.name
                if dst_sub.exists():
                    shutil.rmtree(dst_sub)
                shutil.copytree(f, dst_sub)
        print(f"[DRIVE] Saved to: {dest}")
    except ImportError:
        print("[DRIVE] Not in Colab — skipping Drive save.")
    except Exception as e:
        print(f"[DRIVE] Save failed: {e}")


# ---------------------------------------------------------------------------
# Phase 1: QLoRA Fine-Tuning
# ---------------------------------------------------------------------------
def run_training(
    train_path: Path,
    val_path: Optional[Path],
    output_dir: Path,
    model_name: str,
    max_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    lr: float,
) -> None:
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU required for QLoRA training (T4 or higher).")
        sys.exit(1)

    print("=" * 72)
    print(f"  VeriLog AI — Stage B QLoRA Fine-Tuning")
    print(f"  Model:  {model_name}")
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  Steps:  {max_steps}")
    print(f"  Batch:  {batch_size} × {gradient_accumulation_steps} grad accum")
    print("=" * 72)

    # 1. 4-bit quantization (float16 for T4 compatibility)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4 lacks native BF16
    )

    print("\n[1/5] Loading tokenizer and 4-bit model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # 2. LoRA configuration
    print("\n[2/5] Attaching LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Format dataset
    print("\n[3/5] Preparing training data...")
    train_rows = _load_jsonl(train_path)
    _print_label_distribution(train_rows, "Train")

    def _make_formatted_dataset(jsonl_path: Path) -> HFDataset:
        rows = _load_jsonl(jsonl_path)
        texts = []
        for r in rows:
            prompt = format_prompt(r.get("log", ""))
            target = _format_target(r)
            texts.append(f"<s>[INST] {prompt} [/INST]\n{target}</s>")
        return HFDataset.from_dict({"text": texts})

    train_ds = _make_formatted_dataset(train_path)
    print(f"  Training set: {len(train_ds)} samples")

    val_ds = None
    if val_path and val_path.exists():
        val_ds = _make_formatted_dataset(val_path)
        print(f"  Validation set: {len(val_ds)} samples")

    # 4. SFT configuration
    print("\n[4/5] Configuring SFTTrainer...")
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        max_seq_length=2048,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        eval_strategy="epoch" if val_ds else "no",
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        dataset_text_field="text",
        dataset_num_proc=1,
        max_steps=max_steps,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
    )

    # 5. Train & save
    print(f"\n[5/5] Training started (max_steps={max_steps})...")
    train_result = trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    (output_dir / "train_metrics.json").write_text(
        json.dumps(train_result.metrics, indent=2), encoding="utf-8"
    )
    print(f"\n  ✅ LoRA adapters saved to: {output_dir}")

    del trainer, model
    _cleanup_gpu()


# ---------------------------------------------------------------------------
# Phase 2: Merge LoRA into base weights
# ---------------------------------------------------------------------------
def run_merging(model_name: str, adapter_dir: Path, merged_dir: Path) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'=' * 72}")
    print(f"  Merging LoRA adapters into base model")
    print(f"  Base:    {model_name}")
    print(f"  Adapter: {adapter_dir}")
    print(f"{'=' * 72}")

    _cleanup_gpu()

    print("\n[1/3] Loading base model in FP16 to GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("[2/3] Loading and merging PEFT adapters...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()

    print(f"[3/3] Saving merged model to {merged_dir}...")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    del merged_model, peft_model, base_model
    _cleanup_gpu()
    print("  ✅ Merged model saved and memory cleared!")


# ---------------------------------------------------------------------------
# Phase 3: Benchmark inference latency + accuracy
# ---------------------------------------------------------------------------
def run_benchmark(model_dir: Path, test_dataset: Path, report_path: Path) -> None:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    print(f"\n{'=' * 72}")
    print(f"  Latency & Accuracy Benchmark")
    print(f"  Model: {model_dir}")
    print(f"  Test:  {test_dataset}")
    print(f"{'=' * 72}")

    if not test_dataset.exists():
        print(f"[ERROR] Test dataset not found: {test_dataset}")
        sys.exit(1)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("\nLoading model in 4-bit for inference...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # JSON stopping criteria — halt generation at first '}'
    class JSONStop(StoppingCriteria):
        def __init__(self, token_id: int):
            self.token_id = token_id
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kw) -> bool:
            return input_ids[0][-1].item() == self.token_id

    closing_brace_id = tokenizer.convert_tokens_to_ids("}")
    stop_criteria = StoppingCriteriaList([JSONStop(closing_brace_id)])

    test_rows = _load_jsonl(test_dataset)
    print(f"Loaded {len(test_rows)} test samples.\n")

    latencies, speeds, details = [], [], []
    correct = 0

    print(f"{'#':<5} | {'Expected':<28} | {'Predicted':<28} | {'Time':<6} | {'tok/s':<6}")
    print("-" * 80)

    for idx, row in enumerate(test_rows):
        log_text = row.get("log", "")
        expected = row.get("label", "Unknown")

        prompt = f"<s>[INST] {format_prompt(log_text)} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stop_criteria,
                do_sample=False,
            )
        elapsed = time.time() - t0
        latencies.append(elapsed)

        gen_tokens = outputs[0][input_len:]
        n_tok = len(gen_tokens)
        speed = n_tok / elapsed if elapsed > 0 else 0.0
        speeds.append(speed)

        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        predicted = "Unknown"
        try:
            si = gen_text.find("{")
            ei = gen_text.rfind("}")
            if si != -1 and ei != -1 and ei > si:
                parsed = json.loads(gen_text[si:ei+1])
                predicted = parsed.get("label", "Unknown")
        except Exception:
            pass

        is_correct = predicted.lower() == expected.lower()
        if is_correct:
            correct += 1

        details.append({
            "id": idx, "expected": expected, "predicted": predicted,
            "correct": is_correct, "latency_s": round(elapsed, 3),
        })

        print(f"{idx+1:<5} | {expected[:28]:<28} | {predicted[:28]:<28} | {elapsed:.2f}s | {speed:.0f}")

    total = len(test_rows)
    accuracy = correct / total if total else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    print("-" * 80)
    print(f"  Accuracy:  {accuracy * 100:.2f}% ({correct}/{total})")
    print(f"  Avg Latency: {avg_lat:.3f}s")
    print(f"  Avg Speed:   {avg_speed:.0f} tokens/s")

    report = {
        "accuracy": accuracy, "total": total, "correct": correct,
        "avg_latency_s": round(avg_lat, 3),
        "avg_tokens_per_sec": round(avg_speed, 1),
        "details": details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Report saved to: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="VeriLog AI — Stage B: CodeLlama-7B QLoRA training pipeline",
    )
    parser.add_argument("--train", action="store_true", help="Run QLoRA training")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA into base weights")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")
    parser.add_argument("--save-to-drive", action="store_true", help="Save model to Google Drive")

    parser.add_argument("--train-dataset", default="data/splits/train.jsonl")
    parser.add_argument("--val-dataset", default="data/splits/val.jsonl")
    parser.add_argument("--test-dataset", default="data/splits/test.jsonl")
    parser.add_argument("--output-dir", default="artifacts/codellama-adapter")
    parser.add_argument("--merged-dir", default="artifacts/codellama-merged")
    parser.add_argument("--report", default="reports/benchmark_report.json")

    parser.add_argument("--model-name", default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    merged_dir = Path(args.merged_dir)

    if not any([args.train, args.merge, args.benchmark, args.save_to_drive]):
        parser.print_help()
        sys.exit(0)

    if args.train:
        train_path = Path(args.train_dataset)
        if not train_path.exists():
            print(f"[ERROR] Training data not found: {train_path}")
            print("Run: PYTHONPATH=. python -m data.split_dataset")
            sys.exit(1)
        run_training(
            train_path=train_path,
            val_path=Path(args.val_dataset),
            output_dir=output_dir,
            model_name=args.model_name,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            lr=args.lr,
        )

    if args.merge:
        run_merging(args.model_name, output_dir, merged_dir)

    if args.benchmark:
        target = merged_dir if merged_dir.exists() else Path(args.model_name)
        run_benchmark(target, Path(args.test_dataset), Path(args.report))

    if args.save_to_drive:
        target = merged_dir if merged_dir.exists() else output_dir
        if target.exists():
            _save_to_drive(target, "verilog-ai-codellama")
        else:
            print(f"[ERROR] No model found at {target}")


if __name__ == "__main__":
    main()
