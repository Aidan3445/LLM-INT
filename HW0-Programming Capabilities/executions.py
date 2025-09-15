import json
import subprocess
import sys
import os
import re

def load_completions(filename = "completions.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples with completions")
    return data

def run_code_with_timeout(code, timeout = 5):
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            'success': result.returncode == 0,
            'timeout': False
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'timeout': True
        }
    except Exception as e:
        return {
            'success': False,
            'timeout': False
        }

pattern = re.compile(r"^def\s+\w+\s*\(.*?\).+:", re.MULTILINE)
def get_function_def(prompt):  
    match = pattern.search(prompt)
    if match:
        print(match.group())

def evaluate_single_sample(to_evaluate):
    [completions, test, prompt] = to_evaluate
    results = []
    get_function_def(prompt)
    for completion_idx, completion in enumerate(completions):
        try:
            
            full_code = completion + '\n\n' + test
            result = run_code_with_timeout(full_code)
            
            status = "PASS" if result['success'] else "FAIL"
            if result['timeout']:
                status = "TIMEOUT"
            print(f"Completion {completion_idx + 1}: {status}")
            results.append(result)
        except Exception as e:
            result = {
                'success': False,
                'timeout': False
            }
            results.append(result)
            print(f"Completion {completion_idx + 1}: ERROR")
    print("------------------")
    return results

data = load_completions()

all_results = []
for index, e in enumerate(data):
    print(f"Run: {index + 1}")
    all_results.extend(evaluate_single_sample(e))

count = len(all_results)
passes = 0
for result in all_results:
    if result['success']:
        passes += 1

print(passes/count)



