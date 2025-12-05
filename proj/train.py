"""
Training script for fine-tuning Llama 3.2 3B on escape room JSON generation.

Uses HuggingFace Transformers + PEFT (LoRA) to train a local model to generate
escape room game JSONs from theme inputs.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import time
import gc
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    # Dataset paths
    dataset_dir: Path = Path("./dataset")
    train_path: Path = Path("./dataset/train.json")
    val_path: Path = Path("./dataset/val.json")
    
    # Model configuration
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir: Path = Path(f"./fine_tuned_model_{time.time()}")
    
    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 4096  # Adjust based on your JSON sizes
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Wandb configuration
    wandb_project: str = "escape-room-llm"
    wandb_run_name: str = "llama3p2-3b-finetune"
    
    # Other
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


config = Config()


# ============================================================
# Data Loading and Formatting
# ============================================================

def load_json_dataset(path: Path) -> list[dict]:
    """Load a jsond dataset of DSPy Examples and convert to dicts."""
    print(f"  Loading {path}...")
    with open(path, "rb") as f:
        examples = json.load(f)
    
    # Convert DSPy Examples to simple dicts
    data = []
    for ex in examples:
        data.append({
            "theme": ex["theme"],
            "json_output": ex["json_output"],
        })
    
    print(f"  Loaded {len(data)} examples")
    return data


def format_prompt(theme: str, json_output: str = None) -> str:
    """Format a training example as a prompt-completion pair."""
    prompt = f"Generate a complete escape room game in JSON format for the following theme.\n\nTheme: {theme}\n\nJSON:"
    
    if json_output is not None:
        return f"{prompt}\n{json_output}"
    return prompt


def tokenize_dataset(
    data: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """Tokenize the dataset for training."""
    
    def tokenize_fn(examples):
        # Format as prompt + completion
        texts = [
            format_prompt(theme, json_out)
            for theme, json_out in zip(examples["theme"], examples["json_output"])
        ]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Convert to HF Dataset
    dataset = Dataset.from_list(data)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["theme", "json_output"],
        desc="Tokenizing",
    )
    
    return tokenized_dataset


# ============================================================
# Model Setup
# ============================================================

def load_model_and_tokenizer(model_path: str):
    """Load the base model and tokenizer."""
    
    print(f"  Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map=None,
        trust_remote_code=True,
    ).to("cuda")
    
    print(f"  Model loaded: {model.num_parameters():,} parameters")
    
    return model, tokenizer


def apply_lora(model, config: Config):
    """Apply LoRA adapters to the model."""
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  LoRA applied:")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model


# ============================================================
# Training
# ============================================================

def train():
    print("=" * 60)
    print("ESCAPE ROOM JSON GENERATOR - TRAINING")
    print("=" * 60)
    
    # [1/8] Initialize wandb
    print("\n[1/8] Initializing wandb...")
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            "base_model": config.base_model,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "max_seq_length": config.max_seq_length,
        }
    )
    print("[1/8] Wandb initialized")
    
    # [2/8] Load datasets
    print("\n[2/8] Loading datasets...")
    train_data = load_json_dataset(config.train_path)
    val_data = load_json_dataset(config.val_path)
    print(f"[2/8] Datasets loaded (train: {len(train_data)}, val: {len(val_data)})")
    
    # [3/8] Load model and tokenizer
    print("\n[3/8] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.base_model)
    print("[3/8] Model and tokenizer loaded")
    
    # [4/8] Apply LoRA
    print("\n[4/8] Applying LoRA adapters...")
    model = apply_lora(model, config)
    print("[4/8] LoRA applied")
    
    # [5/8] Tokenize datasets
    print("\n[5/8] Tokenizing datasets...")
    train_dataset = tokenize_dataset(train_data, tokenizer, config.max_seq_length)
    val_dataset = tokenize_dataset(val_data, tokenizer, config.max_seq_length)
    print(f"[5/8] Datasets tokenized")
    
    # [6/8] Set up training arguments
    print("\n[6/8] Setting up training arguments...")
    print(f"       Epochs: {config.epochs}")
    print(f"       Batch size: {config.batch_size}")
    print(f"       Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"       Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"       Learning rate: {config.learning_rate}")
    print(f"       Max sequence length: {config.max_seq_length}")
    
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=config.wandb_run_name
        )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    print("[6/8] Training configured")
    
    # [7/8] Train
    print("\n[7/8] Starting training...")
    print("-" * 60)
    trainer.train()
    print("-" * 60)
    print("[7/8] Training complete")
    
    # [8/8] Save model
    print(f"\n[8/8] Saving model to {config.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    print("[8/8] Model saved")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return model, tokenizer


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("STARTING TRAINING PIPELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Base model:    {config.base_model}")
    print(f"  Dataset dir:   {config.dataset_dir}")
    print(f"  Output dir:    {config.output_dir}")
    print(f"  Wandb project: {config.wandb_project}")
    
    # Train
    model, tokenizer = train()
    
    # Finalize wandb
    wandb.finish()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()