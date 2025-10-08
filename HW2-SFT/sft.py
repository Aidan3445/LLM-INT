import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tempfile
import subprocess
import re
import wandb
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.nn import functional as F
import argparse
import time
import math

def load(model_name, **kwargs):
    """Load model, tokenizer, and datasets"""
    
    print("Loading Model")
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = datasets.load_dataset("nuprl/engineering-llm-systems", "mbpp-rkt-correct-executions", split="train")
    print("Loading Complete")
    return (model, tokenizer, train_ds)

    
def init_wandb(learning_rate, epochs, **kwargs):
    """Init Weights & Biases data collection"""
    
    print(f"Runnning with Learning Rate: {learning_rate} and epochs: {epochs}")
    w = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="weinberg-ai-northeastern-university",
        # Set the wandb project where this run will be logged.
        project="supervised_fine_tuning",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": learning_rate,
            "architecture": "Qwen",
            "dataset": "mbpp-rkt-test-problems",
            "epochs": epochs,
        }
    )
    print("W&B setup complete")
    return w
    


def parse_args():
    """Specify and parse the CLI arguments"""
    
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning")
    
    # Required positional argument
    parser.add_argument(
        "model_name", 
        type=str, 
        help="Path or HuggingFace model name (e.g., '/scratch/.../Qwen3-8B-Base' or 'Qwen/Qwen3-1.7B-Base')"
    )
    # Optional arguments
    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=0.0005,
        help="Learning rate for the model, between 1e-6 and 1e-3 (default: 0.0005)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=5,
        help="Number of epochs or iterations through the data (default: 5)"
    )
    
    args = parser.parse_args()
    return {
        "model_name": args.model_name,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs
    }


def run(start):
    args = parse_args()
    (model, tokenizer, train_ds) = load(**args)
    w = init_wandb(**args)
    
    item_count = len(train_ds)

    optimizer = AdamW(model.parameters(), lr=args["learning_rate"])
    model.train()
    
    setup_time = time.time() - start_time
    print("Beginning Training") 
    
    try:
        running_loss = 0
        for epoch in tqdm(range(args["epochs"])):
            for index in range(item_count):
                row = train_ds[index]
                description = row["description"]
                input_format = row.get("input_format")
                output_format = row.get("output_format")
                target = row.get("code")
                prompt = f"#lang racket\n;; {description}\n;; input_format\n;; {output_format}\n"
                full_text = prompt + target
                
                gen_start = time.time()
                inputs = tokenizer([full_text], return_tensors="pt").to(model.device)
                labels = inputs.input_ids.clone()
                # label only the result
                labels[:, :len(tokenizer(prompt).input_ids)] = -100  
                
                logits = model.forward(**inputs, labels=labels).logits
                loss = F.cross_entropy(
                    logits[0, :-1], 
                    inputs.input_ids[0, 1:], 
                    ignore_index=-100
                ).cpu()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                running_loss += loss
                avg_loss = running_loss / ((epoch * item_count) + index + 1)

                w.log({ 
                    "loss": loss, 
                    "tokens": logits.shape[1],
                    "duration": time.time() - gen_start,
                    "epoch": epoch,
                    "step": index,
                    "global_step": (epoch * item_count) + index,
                    "lr": args["learning_rate"],
                    "avg_loss": avg_loss,
                    "perplexity": math.exp(avg_loss)
                })

            model.save_pretrained(f"sft-model-lr{args["learning_rate"]}-epoch{epoch}")
            tokenizer.save_pretrained(f"sft-tokenizer-lr{args["learning_rate"]}-epoch{epoch}")
    except Exception as e:
        print(f"ERROR:\n{e}")
    finally:
        total_time = time.time() - start_time
        w.finish()
        print(f"Setup time: {setup_time}\nRun time: {total_time - setup_time}\nTotal time: {total_time}")
    

if __name__ == "__main__":
    start_time = time.time()
    run(start_time)
