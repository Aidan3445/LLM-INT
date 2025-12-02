import argparse
import json
import os

from json_generator import generate_game_json
from compiler import compile_json_to_textworld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Natural-language prompt describing the desired escape-room")
    parser.add_argument("--out-json", default="generated_game.json", help="Output JSON filename")
    parser.add_argument("--out-py", default=None, help="Output Python generator filename (under games/<name>/)")
    args = parser.parse_args()

    game_obj = generate_game_json(args.prompt, provider=args.provider)

    with open(args.out_json, "w") as fh:
        json.dump(game_obj, fh, indent=2)

    print(f"Wrote game JSON to {args.out_json}")

    base_name = os.path.splitext(os.path.basename(args.out_json))[0]
    out_py = args.out_py or f"games/{base_name}/{base_name}_game.py"

    out_dir = os.path.dirname(out_py)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Compiling JSON -> TextWorld Python: {out_py}")
    compile_json_to_textworld(args.out_json, out_py)
    print("Done.")


if __name__ == "__main__":
    main()
