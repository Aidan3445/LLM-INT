import json
import subprocess
import tempfile
from pathlib import Path

def remove_extra_lang_tags(text):
    lines = text.splitlines()
    if lines and lines[0].strip() == "#lang racket":
        lines = lines[1:]
    if lines and lines[-1].strip() == "#lang racket":
        lines = lines[:-1]
    return "\n".join(lines)

def run_racket_code(code, test_input, timeout=5):
    with tempfile.NamedTemporaryFile("w", suffix=".rkt", delete=False, encoding="utf-8") as tmp:
        fixed_code = remove_extra_lang_tags(code)
        tmp.write(fixed_code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            ["racket", tmp_path],
            input=test_input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            timeout=timeout,
            check=False,
        )
        output = proc.stdout.strip()
        print("STDERR:", proc.stderr)
    except subprocess.TimeoutExpired:
        output = "__timeout__"
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return output

def test_solutions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_passes = 0
    total_runs = 0

    for sol_key, sol_val in data.items():
        code = sol_val.get("solution", "")
        tests = sol_val.get("tests", [])

        print(f"Testing solution: {sol_key}")
        for i, test in enumerate(tests):
            inp = test.get("input", "")
            expected = test.get("output", "").strip()
            out = run_racket_code(code, inp)
            pass_test = (out == expected)
            if pass_test:
                total_passes += 1
            status = "PASS" if pass_test else "FAIL"
            print(f" Test {i}: {status}")
            if not pass_test:
                print(f"  Input:\n{inp}")
                print(f"  Expected:\n{expected}")
                print(f"  Got:\n{out}")
            total_runs += 1
    
    print(f"TOTAL PASSES: {total_passes} OUT OF {total_runs} TOTAL RUNS")
    print(f"TOTAL PASS RATE: {total_passes / total_runs}")

test_solutions("lr0.0005-epoch4_solutions.json")

