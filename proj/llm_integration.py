import dspy
from dspy import LM
from pathlib import Path
import json

class LLM_intercepter:
    def __init__(self, model="claude-sonnet-4-5", api_key="", api_base="https://api.litellm.ai"):
        self.llm = LM(
            model=model,
            api_key=api_key,
            api_base=api_base
        )

        dspy.settings.configure(lm=self.llm)

    def llm_feedback(self, feedback, user_input="", game_json=""):
            """Use an LLM to generate on-theme messages for the user"""
            
            feedback_sig = dspy.Signature("feedback: str, user_input: str, game_json: str -> enhanced_feedback: str")
            
            feedback_predict = dspy.Predict(feedback_sig)
            feedback_predict.set_lm(self.llm)
            return feedback_predict(feedback=feedback, user_input=user_input, game_json=game_json)
        
        
if __name__ == "__main__":
    print('starting')
    # Instantiate the interceptor
    interceptor = LLM_intercepter(
        model="claude-sonnet-4-5",
        api_key="sk-D2aaKxoUVvyW2iCsw5-ULg",
        api_base="https://api.litellm.ai"
    )
    print('set interceptor')

    # Example inputs
    original_feedback = """-= Tower Bedroom =-
You've just sauntered into a Tower Bedroom.

You can make out a wooden closet. You wonder idly who left that here. You see a top drawer. You see a middle drawer, so there's that. You make out a closed bottom drawer.

There is a closed wooden trapdoor leading south.

There is a four-poster bed and an oak dresser on the floor."""
    user_input = "take key"
    game_path = Path("games/example_1/game.json")
    with open(game_path, "r") as f:
        game_json = json.dumps(json.load(f))  # convert to string for llm_feedback
    print('loaded json')
    # Call the llm_feedback method
    enhanced_feedback = interceptor.llm_feedback(
        feedback=original_feedback,
        user_input=user_input,
        game_json=game_json
    )

    # Print the output
    print("Original Feedback:")
    print(original_feedback)
    print("\nEnhanced Feedback:")
    print(enhanced_feedback)