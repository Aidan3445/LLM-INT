import json
import os
import gc
import torch
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy
from transformers import AutoModelForCausalLM, AutoTokenizer
from compiler import compile_json_to_textworld
import time


# ============================================================
# CONFIGURATION
# ============================================================

EXAMPLE_THEMES = [
    "Haunted Victorian Library",
    "Abandoned Space Station",
    "Underwater Research Lab"
]

MAX_THEME_WORDS = 10
MAX_THEME_CHARS = 40
NUM_PARALLEL_GENERATIONS = 2
MAX_NEW_TOKENS = 3000
GAME_DIR = Path("games")


# ============================================================
# DSPy Setup
# ============================================================

class EscapeRoomGenerator(dspy.Signature):
    """Generate a complete escape room game in JSON format."""
    
    theme = dspy.InputField(desc="The theme for the escape room")
    json_output = dspy.OutputField(desc="Complete escape room JSON configuration")


class HuggingFaceModel(dspy.LM):
    """DSPy wrapper for local HuggingFace model."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model from {self.model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
        )
        self.model = self.model.to("cuda")
        self.model.eval()
        print(f"Model loaded: {self.model.num_parameters():,} parameters")
    
    def __call__(self, prompt: str, **kwargs) -> list[str]:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_tokens', MAX_NEW_TOKENS),
                do_sample=True,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [generated]


# ============================================================
# Formatting
# ============================================================

def format_prompt(theme: str) -> str:
    """Format a theme into a generation prompt."""
    return f"Generate a complete escape room game in JSON format for the following theme.\n\nTheme: {theme}\n\nJSON:"


def extract_json_from_generation(generated: str) -> str:
    """Extract JSON content from generated text."""
    # Find the JSON section
    first_clip = generated.find("JSON:")
    if first_clip != -1:
        generated = generated[first_clip + 5:]
    
    # Extract from first { to last }
    start = generated.find("{")
    if start == -1:
        raise ValueError("No opening brace found in generated text")
    
    generated = generated[start:]
    
    # Remove any text after a second "JSON:" marker
    second_clip = generated.find("JSON:")
    if second_clip != -1:
        generated = generated[:second_clip]
    
    end = generated.rfind("}") + 1
    if end == 0:
        raise ValueError("No closing brace found in generated text")
    
    return generated[:end]


# ============================================================
# Generation & Compilation
# ============================================================

def generate_single_game(lm: HuggingFaceModel, theme: str, attempt_num: int) -> tuple[int, str, Exception]:
    """Generate a single game JSON. Returns (attempt_num, json_str, error)."""
    try:
        prompt = format_prompt(theme)
        generated = lm(prompt)[0]
        json_str = extract_json_from_generation(generated)
        
        # Validate JSON
        json.loads(json_str)
        
        return (attempt_num, json_str, None)
    except Exception as e:
        return (attempt_num, None, e)


def generate_games_parallel(lm: HuggingFaceModel, theme: str) -> list[tuple[int, str, Exception]]:
    """Generate multiple games in parallel."""
    if NUM_PARALLEL_GENERATIONS <= 0:
        return []
    if NUM_PARALLEL_GENERATIONS == 1:
        print("\nGenerating 1 game variation...")
    else:
        print(f"\nGenerating {NUM_PARALLEL_GENERATIONS} game variations in parallel...")
    print("This may take a minute or two. Please be patient!\n")
    
    results = []
    with ThreadPoolExecutor(max_workers=NUM_PARALLEL_GENERATIONS) as executor:
        start = time.time()
        futures = {
            executor.submit(generate_single_game, lm, theme, i): i 
            for i in range(NUM_PARALLEL_GENERATIONS)
        }
        
        for future in as_completed(futures):
            attempt_num, json_str, error = future.result()
            results.append((attempt_num, json_str, error))
            
            if error is None:
                print(f"  ✓ Generation {attempt_num + 1}: Valid JSON generated")
            else:
                print(f"  ✗ Generation {attempt_num + 1}: Failed - {type(error).__name__}")
        elapsed = time.time() - start
        print(f"\nGeneration{"s" if NUM_PARALLEL_GENERATIONS > 1 else ""} completed in {elapsed:.2f} seconds.")
    
    # Sort by attempt number
    results.sort(key=lambda x: x[0])
    return results


def try_compile_and_run(json_str: str, json_path: str) -> bool:
    """Try to compile JSON to TextWorld and return success status."""
    json_path = Path(json_path)
    py_path = json_path.with_suffix('.py')
    
    print(f"attempting compile {json_path} to {py_path}")
    try:
        # Ensure parent directory exists
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(json_path, 'w') as f:
            f.write(json_str)
            print(f"wrote json to {json_path}")
        
        compile_json_to_textworld(str(json_path), str(py_path))
        
        # Clean up temp py file
        if py_path.exists():
            os.remove(py_path)

        return True
    except Exception as e:
        print(f"  ✗ Compilation failed: {e}")
        return False


# ============================================================
# User Interaction
# ============================================================

def get_theme_from_user() -> str:
    """Prompt user for a theme with validation."""
    print("\n" + "=" * 60)
    print("ESCAPE ROOM THEME SELECTOR")
    print("=" * 60)
    print("\nExample themes:")
    for i, theme in enumerate(EXAMPLE_THEMES, 1):
        print(f"  {i}. {theme}")
    print()
    
    while True:
        theme = input("Enter your theme (or 'quit' to exit): ").strip()
        
        if theme.lower() == 'quit':
            return None
        
        if not theme:
            print("Theme cannot be empty. Please try again.\n")
            continue
        
        word_count = len(theme.split())
        char_count = len(theme)
        
        if word_count > MAX_THEME_WORDS:
            print(f"Warning: Theme has {word_count} words (recommended: ≤{MAX_THEME_WORDS})")
            print("   Longer themes may produce less coherent results.")
            confirm = input("   Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        
        if char_count > MAX_THEME_CHARS:
            print(f"Warning: Theme has {char_count} characters (recommended: ≤{MAX_THEME_CHARS})")
            print("   Longer themes may produce less coherent results.")
            confirm = input("   Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        
        return theme


def ask_retry() -> str:
    """Ask user if they want to retry or exit."""
    print("\n" + "-" * 60)
    print("What would you like to do?")
    print("  1. Try again with the same theme")
    print("  2. Try a different theme")
    print("  3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


# ============================================================
# Main Game Loop
# ============================================================

def main(model_path: str):
    """Main game loop."""
    # Ensure temp directory exists
    GAME_DIR.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("ESCAPE ROOM GAME GENERATOR")
    print("=" * 60)
    print("\nInitializing...")
    
    # Load model
    print("\n[1/2] Loading model...")
    lm = HuggingFaceModel(model_path)
    print("[1/2] ✓ Model loaded successfully")
    
    print("\n[2/2] Setting up DSPy...")
    dspy.settings.configure(lm=lm)
    print("[2/2] ✓ DSPy configured")
    
    current_theme = None
    
    while True:
        # Get theme from user
        if current_theme is None:
            theme = get_theme_from_user()
            if theme is None:
                print("\nThanks for playing! Goodbye!")
                break
            current_theme = theme
        else:
            theme = current_theme
        
        print(f"Generating games with theme: {theme}")
        
        # Generate games in parallel
        results = generate_games_parallel(lm, theme)
        
        # Try each generation
        success = False
        game_id = current_theme.lower().replace(" ", "_")
        # Where to save generated JSON
        generated_json_path = GAME_DIR / f"generated/{game_id}.json"
        # Paths for compiled game data
        json_path = GAME_DIR / f"{game_id}.json"
        py_path = GAME_DIR / f"{game_id}.py"
        
        print(f"\nTesting generated games...")
        
        for attempt_num, json_str, gen_error in results:
            if gen_error is not None:
                print(f"\n  Generation {attempt_num + 1}: Skipped (generation failed)")
                print(f"    Error: {gen_error}")
                continue
            
            print(f"\n  Testing Generation {attempt_num + 1}...")
            
            # Try to compile
            if try_compile_and_run(json_str, generated_json_path):
                print(f"    ✓ Compilation successful!")
                success = True
                break
        
        if success:
            # Launch gameplay
            print("\n" + "=" * 60)
            print("LAUNCHING GAME...")
            print("=" * 60)
            print("\n")
            
            try:
                result = subprocess.run(
                    [sys.executable, "gameplay_interface.py", str(generated_json_path), "--force"],
                    check=True
                )
                print("\n" + "=" * 60)
                print("GAME COMPLETED!")
                print("=" * 60)
                current_theme = None  # Reset theme for next iteration
            except subprocess.CalledProcessError as e:
                print(f"\nGameplay error: {e}")
                current_theme = None
            except KeyboardInterrupt:
                print("\n\nGame interrupted by user")
                current_theme = None
        else:
            # All generations failed
            print("\n" + "=" * 60)
            print("ALL GENERATIONS FAILED")
            print("=" * 60)
            print("\nFailure summary:")
            for attempt_num, json_str, gen_error in results:
                if gen_error is not None:
                    print(f"  Generation {attempt_num + 1}: {type(gen_error).__name__} - {gen_error}")
                else:
                    print(f"  Generation {attempt_num + 1}: Compilation failed")
            
            # Ask what to do next
            choice = ask_retry()
            if choice == '1':
                # Keep current theme
                continue
            elif choice == '2':
                # Get new theme
                current_theme = None
                continue
            else:
                # Exit
                print("\nThanks for playing! Goodbye!")
                break
        
        # Clean up temp files
        if json_path.exists():
            json_path.unlink()
        if py_path.exists():
            py_path.unlink()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive escape room game generator")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory"
    )
    
    args = parser.parse_args()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        main(args.model_path)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        print("Goodbye!")
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
