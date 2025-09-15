from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

ds = load_dataset("nuprl/engineering-llm-systems", "humaneval", split = "test")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base").to("mps")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

def clip_completions(completion, clip_at = ["\ndef", "\nclass", "\nif", "\nprint", "<|endoftext|>", "```\n\n"]):
    result = completion
    for clip in clip_at:
        result = result.split(clip)[0]
    return result    

prompt_examples = '''Instruction:
Write a function defintion based on the specification, any `import` lines from the input should be included in the output

Input:
```
def strlen(string: str) -> int:
    """ Return length of given string
    >>> strlen('')
    0
    >>> strlen('abc')
    3
    """
```
    
Output:
```
def strlen(string: str) -> int:
    return len(string)
```


Instruction:
Write a function defintion based on the specification, any `import` lines from the input should be included in the output

Input:
```
def add(x: int, y: int) -> int:
    """Add two numbers x and y
    >>> add(2, 3)
    5
    >>> add(5, 7)
    12
    """
```

Output:
```
def add(x: int, y: int) -> int:
    return x + y
```


Instruction:
Write a function defintion based on the specification, any `import` lines from the input should be included in the output

Input:
```
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
```

Output:
```
def fibfib(n: int) -> int:
    if n == 0:
        return 0
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibfib(n - 1) + fibfib(n - 2) + fibfib(n - 3)
```


Instruction:
Write a function defintion based on the specification, any `import` lines from the input should be included in the output

Input:
```
def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct characters (regardless of case) does it consist of
    >>> count_distinct_characters('xyzXYZ')
    3
    >>> count_distinct_characters('Jerry')
    4
    """
```

Output:
```
def count_distinct_characters(string: str) -> int:
    return len(set(string.lower()))
```
'''

def make_prompt(sample):
    return f"""{prompt_examples}


Instruction:
Write a function defintion based on the specification, any `import` lines from the input should be included in the output

Input:
```
{sample["prompt"]}
```

Output:
```
"""

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
