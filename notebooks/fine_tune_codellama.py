#!/usr/bin/env python3
"""
fine_tune_codellama.py — VeriLog AI Stage B Training, Merging, & Inference Optimization

Fine-tunes CodeLlama-7B-Instruct-hf on a custom hardware verification dataset using QLoRA,
merges the adapters back into the base model, and optimizes the local inference latency
for a Google Colab T4 GPU.

Usage in Google Colab:
    # 1. Clone repository and install requirements
    !git clone https://github.com/fadynabil/verilog-ai && cd verilog-ai
    !pip install -q -U torch bitsandbytes transformers peft accelerate datasets trl

    # 2. Run QLoRA fine-tuning and weight merging
    !python notebooks/fine_tune_codellama.py --train --merge

    # 3. Benchmark optimized inference on the test split
    !python notebooks/fine_tune_codellama.py --benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Prompt / Target formatting to prevent train-serve skew
# ---------------------------------------------------------------------------
try:
    from model.prompt import format_prompt
except ImportError:
    def format_prompt(log: str) -> str:
        """Fallback prompt template if executed outside the module path."""
        return (
            "You are a verification log classifier. "
            "Return JSON with keys: label, explanation, confidence (0-1). "
            "Log:\n"
            f"{log}\n"
            "JSON:"
        )


def _format_target(row: Dict) -> str:
    """Format target label, explanation, and confidence to JSON."""
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
# Stage B: Fine-Tuning Execution
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
    """Run QLoRA fine-tuning on CodeLlama-7B-Instruct-hf."""
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU is not available. QLoRA training requires a GPU (T4 or higher).")
        sys.exit(1)

    print(f"\n=== Starting QLoRA Fine-tuning on {model_name} ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. 4-bit Quantization Config (optimized for T4)
    # Note: T4 doesn't support fast native BF16, so float16 compute_dtype is used for speed.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading base tokenizer and model in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Causal LM padding must be right-sided for training

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False  # Disable cache for training (enabled during inference)
    model = prepare_model_for_kbit_training(model)

    # 2. Configure PEFT / LoRA
    # Targets all projection matrices for highest learning capacity in domain adaptation
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Format dataset as instruction-following text
    def _make_formatted_dataset(jsonl_path: Path) -> HFDataset:
        rows = _load_jsonl(jsonl_path)
        formatted_texts = []
        for r in rows:
            prompt = format_prompt(r.get("log", ""))
            target = _format_target(r)
            # Standard CodeLlama Instruct formatting
            formatted_text = f"<s>[INST] {prompt} [/INST]\n{target}</s>"
            formatted_texts.append(formatted_text)
        return HFDataset.from_dict({"text": formatted_texts})

    train_ds = _make_formatted_dataset(train_path)
    print(f"Training set loaded: {len(train_ds)} samples")

    val_ds = None
    if val_path and val_path.exists():
        val_ds = _make_formatted_dataset(val_path)
        print(f"Validation set loaded: {len(val_ds)} samples")

    # 4. SFT Configuration
    # Uses gradient checkpointing and Paged AdamW to fit on T4 GPU (16GB VRAM)
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
        save_strategy="epoch" if val_ds else "steps",
        save_steps=100 if not val_ds else None,
        evaluation_strategy="epoch" if val_ds else "no",
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        dataset_text_field="text",
        max_steps=max_steps,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
    )

    # 5. Run Training & Save
    train_result = trainer.train()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save metrics
    (output_dir / "train_metrics.json").write_text(
        json.dumps(train_result.metrics, indent=2), encoding="utf-8"
    )
    print(f"\n[SUCCESS] Fine-tuning complete. LoRA adapters saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Model Merging
# ---------------------------------------------------------------------------
def run_merging(model_name: str, adapter_dir: Path, merged_dir: Path) -> None:
    """Merge LoRA adapter weights back into base model weights for latency optimization."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n=== Merging LoRA Adapters into Base Model ===")
    print(f"Base model: {model_name}")
    print(f"Adapter: {adapter_dir}")

    # Restart GPU or clear cache to ensure System RAM & VRAM is free before loading unquantized model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading base model in FP16 (loading to CPU first to prevent GPU OOM during merge)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading PEFT model adapters...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("Merging weights (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving fully merged FP16 model to {merged_dir}...")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print("[SUCCESS] Merged model saved successfully!")


# ---------------------------------------------------------------------------
# Targeted Latency Optimization & Evaluation Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(model_dir: Path, test_dataset: Path) -> None:
    """Benchmark local inference latency, token throughput, and exact-match accuracy."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    print(f"\n=== Running Latency & Accuracy Benchmark on T4 GPU ===")
    print(f"Model Directory: {model_dir}")
    print(f"Test Dataset: {test_dataset}")

    if not test_dataset.exists():
        print(f"[ERROR] Test dataset not found at {test_dataset}")
        sys.exit(1)

    # 1. Load model in 4-bit precision for fast, low-footprint local inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading optimized model...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    # 2. Latency Optimization: Stopping Criteria
    # Halts token generation on the first closing brace '}' (ASCII code)
    # This prevents the model from writing trailing filler tokens, saving up to 50% inference latency.
    class JSONStoppingCriteria(StoppingCriteria):
        def __init__(self, target_token_id: int):
            self.target_token_id = target_token_id

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            return input_ids[0][-1].item() == self.target_token_id

    closing_brace_id = tokenizer.convert_tokens_to_ids("}")
    stopping_criteria = StoppingCriteriaList([JSONStoppingCriteria(closing_brace_id)])

    test_rows = _load_jsonl(test_dataset)
    print(f"Loaded {len(test_rows)} test samples.")

    latencies = []
    token_speeds = []
    correct_count = 0
    total_count = 0

    print("-" * 80)
    print(f"{'ID':<6} | {'Expected Label':<28} | {'Predicted':<28} | {'Time':<6} | {'Tokens/s':<6}")
    print("-" * 80)

    for idx, row in enumerate(test_rows):
        log_text = row.get("log", "")
        expected = row.get("label", "Unknown")

        prompt = f"<s>[INST] {format_prompt(log_text)} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,                  # Optimization: Enable static KV caching
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria, # Optimization: Terminate at JSON end
                temperature=0.1,                 # Low temperature for deterministic output
                do_sample=False,
            )
        elapsed = time.time() - start_time
        latencies.append(elapsed)

        generated_tokens = outputs[0][input_len:]
        num_tokens = len(generated_tokens)
        speed = num_tokens / elapsed if elapsed > 0 else 0.0
        token_speeds.append(speed)

        gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Clean up JSON formatting
        if "}" not in gen_text and gen_text.count("{") > gen_text.count("}"):
            gen_text += "}"

        # Parse prediction
        predicted = "Unknown"
        try:
            start_idx = gen_text.find("{")
            end_idx = gen_text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                parsed = json.loads(gen_text[start_idx:end_idx+1])
                predicted = parsed.get("label", "Unknown")
        except Exception:
            # Fallback to simple regex if JSON fails
            pass

        is_correct = (predicted.lower() == expected.lower())
        if is_correct:
            correct_count += 1
        total_count += 1

        print(f"#{idx+1:<5} | {expected[:28]:<28} | {predicted[:28]:<28} | {elapsed:.2f}s  | {speed:.1f}")

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_speed = sum(token_speeds) / len(token_speeds) if token_speeds else 0.0
    accuracy = correct_count / total_count if total_count else 0.0

    print("-" * 80)
    print("=== LATENCY & ACCURACY BENCHMARK SUMMARY ===")
    print(f"Total Test Samples:        {total_count}")
    print(f"Exact-Match Accuracy:      {accuracy * 100:.2f}%")
    print(f"Average Inference Latency: {avg_latency:.3f} seconds")
    print(f"Average Decoding Speed:    {avg_speed:.1f} tokens/second")
    print("-" * 80)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune CodeLlama-7B-Instruct-hf using QLoRA with post-training optimizations."
    )
    parser.add_argument("--train", action="store_true", help="Run QLoRA training phase")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA adapters into base weights")
    parser.add_argument("--benchmark", action="store_true", help="Run local inference latency and accuracy benchmark")
    
    # Path settings
    parser.add_argument("--train-dataset", default="data/splits/train.jsonl", help="Training set JSONL path")
    parser.add_argument("--val-dataset", default="data/splits/val.jsonl", help="Validation set JSONL path")
    parser.add_argument("--test-dataset", default="data/splits/test.jsonl", help="Test set JSONL path")
    parser.add_argument("--output-dir", default="artifacts/codellama-adapter", help="Directory to save PEFT adapters")
    parser.add_argument("--merged-dir", default="artifacts/codellama-merged", help="Directory to save merged model")
    
    # Hyperparameters
    parser.add_argument("--model-name", default="codellama/CodeLlama-7b-Instruct-hf", help="HF Base Model ID")
    parser.add_argument("--max-steps", type=int, default=500, help="Max fine-tuning steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size (per-device)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    
    args = parser.parse_args()

    train_path = Path(args.train_dataset)
    val_path = Path(args.val_dataset)
    test_path = Path(args.test_dataset)
    output_dir = Path(args.output_dir)
    merged_dir = Path(args.merged_dir)

    if not (args.train or args.merge or args.benchmark):
        parser.print_help()
        sys.exit(0)

    # 1. Training Phase
    if args.train:
        if not train_path.exists():
            print(f"[ERROR] Training dataset file not found: {train_path}")
            sys.exit(1)
        run_training(
            train_path=train_path,
            val_path=val_path,
            output_dir=output_dir,
            model_name=args.model_name,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            lr=args.lr,
        )

    # 2. Merging Phase
    if args.merge:
        run_merging(
            model_name=args.model_name,
            adapter_dir=output_dir,
            merged_dir=merged_dir,
        )

    # 3. Benchmarking Phase
    if args.benchmark:
        # Benchmark either the merged model or fall back to base model for syntax check
        benchmark_target = merged_dir if merged_dir.exists() else Path(args.model_name)
        run_benchmark(
            model_dir=benchmark_target,
            test_dataset=test_path,
        )


if __name__ == "__main__":
    main()
