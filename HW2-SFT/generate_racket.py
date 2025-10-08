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
import json
from pathlib import Path
import re

model_name = "lr2e-05-epoch2"

model = AutoModelForCausalLM.from_pretrained(
        f"/scratch/bchk/aweinberg/saved/sft-model-{model_name}",
        torch_dtype=torch.bfloat16
    ).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(f"/scratch/bchk/aweinberg/saved/sft-tokenizer-{model_name}")

test_ds = datasets.load_dataset("nuprl/engineering-llm-systems", "mbpp-rkt-test-problems", split="train")

def clip_generated_code(code):
    """Clip Racket code after the first full program: 
    all top-level balanced forms up through the first call
    to the initially defined function (second appearance of its name)."""

    paren_stack = 0
    last_balanced_idx = 0
    for i, char in enumerate(code):
        if char == '(':
            paren_stack += 1
        elif char == ')':
            paren_stack -= 1
            if paren_stack == 0:
                last_balanced_idx = i + 1
    code = code[:last_balanced_idx]

    func_defs = re.findall(r'\(define\s+\((\w+)', code)
    if not func_defs:
        return code.strip()
    main_func = func_defs[0]
    
    pattern_comment_call = rf';;\s*\((?:display|printf|print|write)?\s*\(?{main_func}\b[^\)]*\)?\)'
    code = re.sub(
        pattern_comment_call,
        lambda m: m.group(0).lstrip(';').strip(),
        code
    )

    matches = list(re.finditer(rf'\({main_func}\b', code))
    if len(matches) >= 2:
        end_idx = matches[1].end()
        paren_stack = 1
        for i in range(end_idx, len(code)):
            if code[i] == '(':
                paren_stack += 1
            elif code[i] == ')':
                paren_stack -= 1
                if paren_stack == 0:
                    end_idx = i + 1
                    break
        code = code[:end_idx]

    display_match = re.search(
        rf'\((?:display|printf|print|write)\b[^\)]*\({main_func}\b[^\)]*\)\)',
        code
    )
    if display_match:
        code = code[:display_match.end()]

    return code.strip()

def generate_text_batch(prompt, model, tokenizer, num_samples=5):
    """Generate using provided prompt, model, and tokenizer
    will generate a specified number of samples for each prompt"""
    
    input = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **input,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=500,
        num_return_sequences=num_samples,
        top_p=.95
    )
    results = []
    for i in range(num_samples):
        output = tokenizer.decode(outputs[i])
        if output.startswith(prompt):
            generated_part = output[len(prompt):]
        else:
            generated_part = output
        generated_part = generated_part.split("<|endoftext|>")[0]
        clipped_generation = clip_generated_code(generated_part)
        results.append((prompt + clipped_generation, generated_part))
    return results

def write_solutions_to_file(solutions, input_format, output_format, tests, filepath="solutions.json"):
    path = Path(filepath)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    start_idx = len(data) + 1
    for i, (sol, unclipped) in enumerate(solutions, start=start_idx):
        key = f"solution #{i}"
        data[key] = {
            "solution": sol,
            "unclipped": unclipped,
            "inputs": input_format,
            "outputs": output_format,
            "tests": tests
        }

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Written {len(solutions)} solutions to {filepath}")
    
def run():
    for i in range(len(test_ds)):
        row = test_ds[i]
        description = row["description"]
        input_format = row.get("input_format")
        output_format = row.get("output_format")
        tests = row.get("tests")
        target = row.get("code")
        prompt = f"""#lang racket
;; {description}
;; {input_format}
;; {output_format}"""

        solutions = generate_text_batch(prompt, model, tokenizer, num_samples=1)

        for solution in solutions:
            write_solutions_to_file(solutions, input_format, output_format, tests, filepath=f"{model_name}_solutions.json")
            
if __name__ == "__main__":
    run()