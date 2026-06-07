from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


from model.prompt import format_prompt


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


def _is_seq2seq_model(model_name: str) -> bool:
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(model_name)
        return getattr(config, "is_encoder_decoder", False)
    except Exception:
        # Fallback based on common names if offline or error
        name_lower = model_name.lower()
        if "t5" in name_lower or "bart" in name_lower or "pegasus" in name_lower:
            return True
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a seq2seq or causal model for verification log classification"
    )
    parser.add_argument("--dataset", default="data/dataset.jsonl", help="Training JSONL")
    parser.add_argument("--validation", default=None, help="Optional validation JSONL")
    parser.add_argument(
        "--model-name",
        default="google/flan-t5-small",
        help="HF model ID",
    )
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "seq2seq", "causal"],
        help="Model type (seq2seq or causal). If 'auto', auto-detected from name/config.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/verilog-ai-model",
        help="Directory for checkpoints and final model",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA adapters (requires peft)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load base model in 4-bit precision (QLoRA) using bitsandbytes",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load base model in 8-bit precision using bitsandbytes",
    )
    args = parser.parse_args()

    # Import heavy deps lazily so other workflows can run without ML stack.
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    rows = _load_jsonl(Path(args.dataset))
    if not rows:
        raise ValueError(f"No rows found in dataset: {args.dataset}")

    if args.model_type == "auto":
        is_seq2seq = _is_seq2seq_model(args.model_name)
    else:
        is_seq2seq = (args.model_type == "seq2seq")

    print(f"Model Name: {args.model_name}")
    print(f"Detected Seq2Seq: {is_seq2seq}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup quantization configurations
    bnb_config = None
    if args.load_in_4bit:
        import torch
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit QLoRA quantization configuration")
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print("Using 8-bit quantization configuration")

    # Load model with quantization if specified
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto" if bnb_config else None,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto" if bnb_config else None,
            low_cpu_mem_usage=True,
        )
        model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA / PEFT setup
    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        
        if args.load_in_4bit or args.load_in_8bit:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)

        if is_seq2seq:
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
        else:
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj"],
            )
        model = get_peft_model(model, lora_cfg)
        print("LoRA configuration applied successfully.")

    ds_rows = [{"prompt": format_prompt(r.get("log", "")), "target": _format_target(r)} for r in rows]
    dataset = Dataset.from_list(ds_rows)

    if is_seq2seq:
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
    else:
        def preprocess(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
            input_ids_list = []
            attention_mask_list = []
            labels_list = []
            for prompt, target in zip(batch["prompt"], batch["target"]):
                prompt_encoded = tokenizer(prompt, truncation=True, max_length=args.max_input_length)
                target_encoded = tokenizer(target, truncation=True, max_length=args.max_target_length)
                
                p_ids = prompt_encoded["input_ids"]
                t_ids = target_encoded["input_ids"]
                
                if len(t_ids) == 0 or t_ids[-1] != tokenizer.eos_token_id:
                    t_ids = t_ids + [tokenizer.eos_token_id]
                    
                input_ids = p_ids + t_ids
                attention_mask = [1] * len(input_ids)
                labels = [-100] * len(p_ids) + t_ids
                
                total_max = args.max_input_length + args.max_target_length
                if len(input_ids) > total_max:
                    input_ids = input_ids[:total_max]
                    attention_mask = attention_mask[:total_max]
                    labels = labels[:total_max]
                    
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                labels_list.append(labels)
                
            return {
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
                "labels": labels_list
            }

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["prompt", "target"])
    
    val_dataset = None
    if args.validation:
        val_rows = _load_jsonl(Path(args.validation))
        if val_rows:
            ds_val_rows = [{"prompt": format_prompt(r.get("log", "")), "target": _format_target(r)} for r in val_rows]
            val_dataset = Dataset.from_list(ds_val_rows).map(preprocess, batched=True, remove_columns=["prompt", "target"])

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=50 if val_dataset else None,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=val_dataset,
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
