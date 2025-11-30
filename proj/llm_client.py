import os
import json
from typing import Any, Dict
import dspy

def generate_game_json(prompt: str) -> Dict[str, Any]:
    
        api_key = os.environ.get("DSPY_API_KEY")
        if not api_key:
            raise RuntimeError("Please set the DSPY_API_KEY environment variable for dspy.")

        Client = getattr(dspy, "Client", None)
        if Client is None:
            raise RuntimeError(
                "Found 'dspy' package but no 'Client' class. Update llm_client.py to use your provider's API."
            )

        client = Client(api_key=api_key)

        for method_name in ("generate", "generate_text", "chat", "completion"):
            method = getattr(client, method_name, None)
            if callable(method):
                resp = method(prompt)

                if isinstance(resp, str):
                    text = resp
                else:
                    text = getattr(resp, "text", None) or getattr(resp, "output", None) or str(resp)

                try:
                    return json.loads(text)
                except Exception as e:
                    raise RuntimeError(
                        "LLM returned non-JSON output. Ensure the model is instructed to output JSON only."
                    ) from e

        raise RuntimeError(
            "Could not find a usable generate/chat method on dspy client.\n"
            "Edit 'llm_client.py' to call your provider's API and return a JSON string."
        )