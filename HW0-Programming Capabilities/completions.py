from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random

ds = load_dataset("nuprl/engineering-llm-systems", "humaneval", split = "test")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base").to("mps")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

def clip_completions(completion, clip_at = ["\n```", "<|endoftext|>", "\ndef", "\nclass", "\nif __name__"]):
    result = completion
    # Clip at the first occurrence of any of the clip_at strings AFTER the first def
    first_def_index = result.find("def")
    if first_def_index != -1:
        clip_indexes = []
        for clip_str in clip_at:
            clip_index = result.find(clip_str, first_def_index + 3)
            if clip_index != -1:
                clip_indexes.append(clip_index)
        if clip_indexes:
            first_clip_index = min(clip_indexes)
            result = result[:first_clip_index]
    return result

prompt_examples = '''def strlen(string: str) -> int:
    """ Return length of given string
    >>> strlen('')
    0
    >>> strlen('abc')
    3
    """
    return len(string)

def add(x: int, y: int) -> int:
    """Add two numbers x and y
    >>> add(2, 3)
    5
    >>> add(5, 7)
    12
    """
    return x + y

def fibfib(n: int) -> int:
    """The FibFib number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    fibfib(0) == 0
    fibfib(1) == 0
    fibfib(2) == 1
    fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3).
    Please write a function to efficiently compute the n-th element of the fibfib number sequence.
    >>> fibfib(1)
    0
    >>> fibfib(5)
    4
    >>> fibfib(8)
    24
    """
    if n == 0:
        return 0
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibfib(n - 1) + fibfib(n - 2) + fibfib(n - 3)

def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct characters (regardless of case) does it consist of
    >>> count_distinct_characters('xyzXYZ')
    3
    >>> count_distinct_characters('Jerry')
    4
    """
    return len(set(string.lower()))

'''

def make_prompt(sample):
    return f"{prompt_examples}{sample['prompt']}"

def generate_completions(sample, count = 5):
    prompt_with_examples = make_prompt(sample)
    prompts = [sample["prompt"]] * count
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=300,
        num_return_sequences=1
    )
    completions = []
    for output in outputs:
        result = tokenizer.decode(output)
        clipped_result = clip_completions(result)
        completions.append(clipped_result)
    return completions

results = []
for index, sample in enumerate(ds):
    ''' Debugging: randomly skip some samples and limit to 10 completions
    if random.random() < 0.75:
        print(f"Skipping sample {index + 1}")
        continue
    if len(results) == 10:
        break
    '''
    completions = generate_completions(sample)
    results.append((completions, sample["tests"], sample["prompt"]))
    print(f"Sample {index + 1} complete")
    if index % 20 == 0:
        with open("completions.json", "w") as f:
            json.dump(results, f)
        print(f"Checkpoint {index + 1} written to completions.json")
        
with open("completions.json", "w") as f:
    json.dump(results, f)

with open("completions.json") as f:
    loaded = json.load(f)
    if len(loaded) == len(results):
        print("successfuly wrote to completions.json")
    else:
        print("write failed")        
