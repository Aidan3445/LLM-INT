"""
Training script for fine-tuning Llama 3.2 3B on escape room JSON generation.

Uses HuggingFace Transformers + PEFT (LoRA) to train a local model to generate
escape room game JSONs from theme inputs.
"""

import json
import os
import gc
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from compiler import compile_json_to_textworld


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    # Model configuration
    trained_model: str
 
    # Wandb configuration
    wandb_project: str = "escape-room-llm"
    wandb_run_name: str = "llama3p2-3b-finetune"


# ============================================================
# Formatting
# ============================================================

def format_prompt(theme: str, json_output: str = None) -> str:
    """Format a training example as a prompt-completion pair."""
    prompt = f"Generate a complete escape room game in JSON format for the following theme.\n\nTheme: {theme}\n\nJSON:"
    
    if json_output is not None:
        return f"{prompt}\n{json_output}"
    return prompt


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
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    print(f"  Model loaded: {model.num_parameters():,} parameters")
    model = model.to("cuda")
    print(f"  Model moved to CUDA")
    
    return model, tokenizer


# ============================================================
# Evaluation
# ============================================================
evaluation_themes = [
    "Sentient Greenhouse Rebellion",
    "Clockwork Carnival Midway",
    "Alchemist's Floating Laboratory",
    "Cybernetic Monastery Archives",
    "Phantom Jazz Club",
    "Titan Mech Hangar Bay",
    "Enchanted Toy Workshop",
    "Volcanic Temple Forge",
    "Rogue Satellite Command Post",
    "Shapeshifter's Den",
    "Celestial Observatory Spire",
    "Subterranean Mushroom Metropolis Metropolis",
    "Arcane Library of Forbidden Tomes",
    "Bioluminescent Coral Citadel",
    "Time-Dilated Research Bunker",
    "Whispering Bamboo Forest Shrine",
    "Dream-Weaving Pillow Atelier",
    "Quantum Chess Tournament Hall",
    "Leviathan Skeleton Shipyard",
    "Mirror-Maze Court of Illusions",
    "Fractal Garden Conservatory",
    "Astral Nomad Caravan",
    "Golem-Powered Mining Outpost",
    "Forgotten Puppet Parliament",
    "Nebula Glass Blowing Studio",
    "Clock-Stopped City Square",
    "Interdimensional Railway Station",
    "Storm-Chaser Sky Dock",
    "Rune-Etched Meteor Crater",
    "Cryptid Tracking Expedition Base",
    "Bio-engineered Orchard of Echoes",
    "Underwater Gargoyle Sanctuary",
    "Cosmic Kart Racetrack",
    "Singing Canyon Excavation Site",
    "Portal-Fragment Junkyard",
    "Druidic Weather Control Hub",
    "Obsidian Dragon Roost",
    "Aurora Ice-Sculptor’s Hall",
    "Desert Mirage Research Tent",
    "Eternal Autumn Village",
    "Feycourt Maskmaker’s Loft",
    "Parasitic Jungle Ruin",
    "Antique Starship Restoration Bay",
    "Holographic Memory Theatre",
    "Psionic Meditation Sphere",
    "Vampire Botanist’s Greenroom",
    "Gravity-Inverted Ballroom",
    "Astral Beast Taming Arena",
    "Lost Algorithm Archive"
]

def evaluate(config):
    """Run evaluation on test set."""
    # [1/8] Initialize wandb
    print("\n[1/8] Initializing wandb...")
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            "trained_model": config.trained_model,
        }
    )
    print("[1/8] Wandb initialized")
    
    # [2/8] Load model and tokenizer
    print("\n[2/8] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.trained_model)
    
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"\nEvaluating on {len(evaluation_themes)} test examples...")
    
    model.eval()
    correct = 0
    
    for i, example in enumerate(evaluation_themes):
        theme = example
        print(f"  [{i+1}/{len(evaluation_themes)}] Testing theme: {theme[:50]}...")
        
        try:
            # Generate
            prompt = format_prompt(theme)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=3000,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode and extract JSON
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Text may contain extra text before/after JSON, so extract JSON part
            # first `{` to last `}`
            first_clip = generated.find("JSON:")
            generated = generated[first_clip + 5:]
            start = generated.find("{")
            generated = generated[start:]
            second_clip = generated.find("JSON:")
            generated = generated[:second_clip]
            end = generated.rfind("}") + 1
            json_part = generated[:end]
            
            # Try to parse as JSON
            json.loads(json_part)
            correct += 0.1 # valid json is the first battle
            print(f"           Valid JSON")

            # create a temporary file for the json
            with open("temp.json", "w") as f:
                f.write(json_part)

            # compile the json to textworld
            compile_json_to_textworld("temp.json", "temp.py")
            # if we got here no errors were raised,
            # so we can add 0.9 to mark the generation as valid (0.1+0.9=1)
            correct += 0.9
        except json.JSONDecodeError:
            print(f"           Invalid JSON")
        except Exception as e:
            print(f"           Compile Error:\n{e}")
        finally:
            if os.path.exists("temp.json"):
                os.remove("temp.json")
            if os.path.exists("temp.py"):
                os.remove("temp.py")
    
    accuracy = correct / len(evaluation_themes)
    print("\n" + "-" * 60)
    print(f"JSON validity accuracy: {accuracy:.2%} ({correct}/{len(evaluation_themes)})")
    print("-" * 60)
    
    wandb.log({"test_json_validity": accuracy})
    return accuracy


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned escape room JSON generation model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory"
    )
    
    args = parser.parse_args()
    
    # Create config from CLI args
    config = Config(
        trained_model=args.model_path,
        wandb_project="escape-room-llm",
        wandb_run_name="llama3p2-3b-eval"
    )
    
    print("\n" + "=" * 60)
    print("STARTING EVALUATION PIPELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Trained model:    {config.trained_model}")
    print(f"  Wandb project: {config.wandb_project}")
    print(f"  Wandb run name: {config.wandb_run_name}")

    evaluate(config)
    
    # Finalize wandb
    wandb.finish()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()