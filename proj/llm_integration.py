import dspy
from dspy import LM
from pathlib import Path
import json

class LLM_intercepter:
    def __init__(self, model="claude-sonnet-4-5", api_key="", api_base="https://litellm.guha-anderson.com"):
        self.llm = LM(
            model=model,
            api_key=api_key,
            api_base=api_base
        )

        dspy.settings.configure(lm=self.llm)

    def llm_feedback(self, feedback, user_input="", game_json=""):
            """Use an LLM to generate on-theme messages for the user"""
            
            feedback_sig = dspy.Signature("feedback: str, user_input: str, game_json: str -> enhanced_feedback: str", 
                                          instructions="Do not give away any hints. Just enhance the story a bit")
            
            feedback_predict = dspy.Predict(feedback_sig)
            feedback_predict.set_lm(self.llm)
            return feedback_predict(feedback=feedback, user_input=user_input, game_json=game_json)["enhanced_feedback"]
        
        
        
    def interpret_user_input(self, user_input="", game_json=""):
        """Use an LLM to generate on-theme messages for the user"""
        
        user_interpret_sig = dspy.Signature("user_input: str, game_json: str -> textworld_command: str", 
                                        instructions="""Change the user input to a command that textworld can read.
                                        Available text world commands:
                                            look:                describe the current room
                                            goal:                print the goal of this game
                                            inventory:           print player's inventory
                                            go <dir>:            move the player north, east, south or west
                                            examine ...:         examine something more closely
                                            eat ...:             eat edible food
                                            open ...:            open a door or a container
                                            close ...:           close a door or a container
                                            drop ...:            drop an object on the floor
                                            take ...:            take an object that is on the floor
                                            put ... on ...:      place an object on a supporter
                                            take ... from ...:   take an object from a container or a supporter
                                            insert ... into ...: place an object into a container
                                            lock ... with ...:   lock a door or a container with a key
                                            unlock ... with ...: unlock a door or a container with a key
                                        """)
        
        user_interpret_predict = dspy.Predict(user_interpret_sig)
        user_interpret_predict.set_lm(self.llm)
        return user_interpret_predict(user_input=user_input, game_json=game_json)["textworld_command"]

        
        
if __name__ == "__main__":
    print('starting')
    # Instantiate the interceptor
    interceptor = LLM_intercepter(
        model="claude-sonnet-4-5",
        api_key="",
        api_base="https://litellm.guha-anderson.com"
    )
    print('set interceptor')

    # Example inputs
    user_input = "eat the cake"
    game_path = Path("games/example_1/game.json")
    with open(game_path, "r") as f:
        game_json = json.dumps(json.load(f))  # convert to string for llm_feedback
    print('loaded json')
    # Call the llm_feedback method
    enhanced_feedback = interceptor.interpret_user_input(
        user_input=user_input,
        game_json=game_json
    )

    print("\nEnhanced Feedback:")
    print(enhanced_feedback)