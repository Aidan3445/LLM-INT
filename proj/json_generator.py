import os
import json
import dspy
import random
import tempfile
from compiler import compile_json_to_textworld
import jsonschema

# need to change configuration
lm = dspy.LM(
        model="claude-sonnet-4-5",
        api_key="sk-cmU9dANhBlbtImdA3FbRdw",
        api_base="https://litellm.guha-anderson.com"
        )
dspy.settings.configure(lm=lm)

class GenerateGameJSON(dspy.Signature):
    theme = dspy.InputField()
    title = dspy.InputField()
    goal = dspy.InputField()
    json_schema = dspy.InputField(desc="JSON schema to follow exactly")
    #error_feedback = dspy.InputField(desc="Previous validation errors to avoid")
    examples = dspy.InputField(desc="Example valid game JSONs showing the correct format")
    output_json = dspy.OutputField(desc="A VALID JSON dictionary matching the required schema exactly.")
    
class GameGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateGameJSON)

    def forward(self, theme, title, goal, schema, examples):
        result = self.generate(
                theme=theme, 
                title=title, 
                goal=goal, 
                json_schema=json.dumps(schema, indent=2), 
                examples=examples)
        text = result.output_json.strip()
        # Text may contain extra text before/after JSON, so extract JSON part
        # first `{` to last `}`
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]


        try:
            data = json.loads(text)
            return data
        except Exception:
            raise ValueError(f"Invalid JSON received:\n{text}")

class BatchGameGenerator:
    def __init__(self, output_dir="game_jsons_and_txts", n=300, schema=None, example_files=None):
        self.output_dir = output_dir
        self.n = n
        self.schema = schema
        self.example_jsons = self.load_examples(example_files) if example_files else []
        os.makedirs(output_dir, exist_ok=True)
        self.generator = GameGenerator()

    def load_examples(self, example_files):
        examples = []
        for filepath in example_files:
            with open(filepath, 'r') as f:
                example_data = json.load(f)
                examples.append(json.dumps(example_data, indent=2))
        
        return "\n\n---EXAMPLE---\n\n".join(examples)
        
    def random_theme(self):
        themes = [
            "Ancient Temple", "Cyberpunk Heist", "Submarine Lab",
            "Alien Zoo", "Medieval Dungeon", "Steampunk Observatory",
            "Dream Archive", "Quantum Prison", "Sunken City",
            "Secret Library", "Lost Research Station", "Time-Traveler's Workshop", 
            "Haunted Carnival", "Bioluminescent Jungle", "Vampire Manor", 
            "Robot Rebellion Factory", "Atlantis Power Core",
            "Pharaoh’s Afterlife Trial", "Witch’s Cottage", "Pocket-Universe Rift",
            "Frozen Research Vault", "Mystic Puppet Theatre", "Ghost Ship Deck",
            "Arcane Alchemist Lab", "Parallel Universe Apartment", "Sky Pirate Airship",
            "Forgotten Clock Tower", "Mutant Greenhouse", "Virtual Reality Malfunction",
            "Cursed Painting Gallery", "Interstellar Train", "Underground Resistance Bunker",
            "Mythical Beast Sanctuary", "Biohazard Quarantine Zone", "Treasure Hunter's Cabin",
            "Dwarven Forge", "Dragon's Hoard Cavern", "Abandoned Theme Park",
            "Shadow Market Bazaar", "Living Museum Exhibit", "Oracle’s Chamber",
            "Nanotech Cleanroom", "Crystal Cavern Nexus", "Wild West Bank Break-In",
            "Storm-Chaser Mobile Lab", "Knight’s Tournament Grounds", "Zombie Outbreak Motel",
            "Golem Foundry", "Hall of Mirrors Maze", "Space Elevator Control Room",
            "Ancient Underworld Gate", "Fairy Realm Crossing", "Wizard Academy Exam",
            "Corporate Espionage Office", "Meteor Mining Colony", "Cursed Pirate Lagoon",
            "Forbidden Forest Ritual Site", "Alien Embassy Negotiation Room", "Lost Monastery",
            "AI-Controlled Smart Home", "Kaiju Alert Command Center", "Hall of Forgotten Toys",
            "Dream Thief's Hideout", "Martian Terraforming Hub", "Tropical Storm Observatory",
            "Gilded Opera House", "Doomsday Cult Sanctuary", "Genetic Splicing Lab",
            "Shadow Assassin Dojo", "Floating Sky Temple", "Hologram Theater",
            "Crystal-Powered Train Station", "Underground Catacombs", "Magic Lantern Workshop",
            "Wizard’s Menagerie", "Astral Projection Chamber", "Secret Speakeasy",
            "Cursed Mine Shaft", "Interdimensional Post Office", "Neon Noir Detective Office",
            "Runaway Nano-Swarm Zone", "Space-Farm Biodome", "Haunted Doll Factory",
            "Chronomancer’s Observatory", "Gladiator’s Arena Tunnels", "Siren's Cove",
            "Gothic Astronomy Tower", "Bio-Android Nursery", "Soviet Listening Station",
            "Retro Arcade Time Loop", "Quantum Maze Core", "Lunar Ruins Dig Site",
            "Cultivated Dream Orchard", "Alien Language Decryption Chamber",
            "Architect’s Impossible House", "Celestial Library", "Meteor Impact Crater Base",
            "Invisible City Marketplace", "Haunted Subway Terminal", "Arcade Hacker Hideout",
            "Eldritch Ritual Chamber", "Magic Carpet Port", "Abandoned Cyber-Zoo",
            "Underground River Temple", "Giant's Kitchen", "Mystery Theater Dressing Room",
            "Undersea Volcano Station", "Ghostly Ballroom"
        ]
        return random.choice(themes)

    def run(self):
        for i in range(8, self.n):
            theme = self.random_theme()
            title = f"{theme} Adventure #{i+1}"
            goal = "Solve the puzzles, achieve the main objective, and escape."

            game = self.generate_valid_game(
                generator=self.generator,
                theme=theme,
                title=title,
                goal=goal,
                schema=self.schema,
                examples=self.example_jsons,
                max_retries=4
            )

            if game is None:
                print(f"✗ FAILED permanently #{i+1} after retries")
                continue

            path = os.path.join(self.output_dir, f"game_{i+1}.json")
            with open(path, "w") as f:
                json.dump(game, f, indent=2)

            print(f"✓ SAVED: {path}")
            
    def generate_valid_game(self, generator, theme, title, goal, schema, examples, max_retries=4):
        for attempt in range(max_retries):
            data = generator(theme=theme, title=title, goal=goal, schema=schema, examples=examples)

            ok, err = self.validate_schema(data, schema)
            if not ok:
                print(f"[Attempt {attempt+1}] Schema error, regenerating: {err}")
                continue

            ok, err = self.validate_compiles(data)
            if not ok:
                print(f"[Attempt {attempt+1}] Compiler error, regenerating:\n{err}")
                continue

            return data

        return None
    

    def validate_compiles(self, json_data, output_dir="compiled_temp"):

        os.makedirs(output_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=output_dir) as tmp_json:
            json.dump(json_data, tmp_json, indent=2)
            json_file = tmp_json.name

        py_file = os.path.splitext(json_file)[0] + ".py"

        try:
            compile_json_to_textworld(json_file, py_file)
            return True, None
        except Exception as e:
            return False, str(e)
        

    def validate_schema(self, json_data, schema):
        try:
            jsonschema.validate(json_data, schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e)
                
                
if __name__ == "__main__":
    GAME_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Adventure Game Configuration",
        "type": "object",
        "required": ["theme", "title", "goal", "rooms", "starting_room"],
        "properties": {
            "theme": {
            "type": "string",
            "description": "The overall theme or setting of the adventure"
            },
            "title": {
            "type": "string",
            "description": "The title of the adventure"
            },
            "goal": {
            "type": "string",
            "description": "Description of the player's objective"
            },
            "starting_room": {
            "type": "string",
            "description": "ID of the room where the player begins"
            },
            "rooms": {
            "type": "array",
            "description": "Array of all rooms in the adventure",
            "minItems": 1,
            "items": {
                "$ref": "#/definitions/room"
            }
            }
        },
        "definitions": {
            "room": {
            "type": "object",
            "required": ["id", "name", "description", "items"],
            "properties": {
                "id": {
                "type": "string",
                "description": "Unique identifier for the room"
                },
                "name": {
                "type": "string",
                "description": "Display name of the room"
                },
                "description": {
                "type": "string",
                "description": "Detailed description of the room"
                },
                "items": {
                "type": "array",
                "description": "Array of items in the room",
                "items": {
                    "$ref": "#/definitions/item"
                }
                },
                "exits": {
                "type": "array",
                "description": "Array of exit objects (if not using door items)",
                "items": {
                    "type": "object"
                }
                },
                "is_goal": {
                "type": "boolean",
                "description": "Whether this room is the goal/winning room"
                },
                "win_message": {
                "type": "string",
                "description": "Message displayed when player reaches this goal room"
                }
            }
            },
            "item": {
            "type": "object",
            "required": ["id", "name", "description"],
            "properties": {
                "id": {
                "type": "string",
                "description": "Unique identifier for the item"
                },
                "name": {
                "type": "string",
                "description": "Display name of the item"
                },
                "description": {
                "type": "string",
                "description": "Description of the item"
                },
                "searchable": {
                "type": "boolean",
                "description": "Whether the item can be searched to find contained items"
                },
                "contains": {
                "type": "array",
                "description": "Array of item IDs contained within this item",
                "items": {
                    "type": "string"
                }
                },
                "subcontainers": {
                "type": "array",
                "description": "Array of sub-containers (e.g., drawers, compartments)",
                "items": {
                    "$ref": "#/definitions/subcontainer"
                }
                },
                "locked": {
                "type": "boolean",
                "description": "Whether the item is locked"
                },
                "lock_type": {
                "type": "string",
                "enum": ["combination", "password", "key"],
                "description": "Type of lock mechanism"
                },
                "key_required": {
                "type": "string",
                "description": "ID of the item needed to unlock this"
                },
                "combination": {
                "type": "string",
                "description": "Combination code to unlock (for combination locks)"
                },
                "password_questions": {
                "type": "array",
                "description": "Array of password questions for password-protected locks",
                "items": {
                    "$ref": "#/definitions/passwordQuestion"
                }
                },
                "readable": {
                "type": "boolean",
                "description": "Whether the item can be read"
                },
                "text": {
                "type": "string",
                "description": "Text content when item is read"
                },
                "leads_to": {
                "type": "string",
                "description": "Room ID this door/exit leads to"
                },
                "direction": {
                "type": "string",
                "description": "Direction name for this exit (e.g., 'north', 'east')"
                },
                "aliases": {
                "type": "array",
                "description": "Alternative names/commands for this item",
                "items": {
                    "type": "string"
                }
                }
            }
            },
            "subcontainer": {
            "type": "object",
            "required": ["id", "name", "locked", "contains"],
            "properties": {
                "id": {
                "type": "string",
                "description": "Unique identifier for the subcontainer"
                },
                "name": {
                "type": "string",
                "description": "Display name of the subcontainer"
                },
                "locked": {
                "type": "boolean",
                "description": "Whether the subcontainer is locked"
                },
                "lock_type": {
                "type": "string",
                "enum": ["combination", "key"],
                "description": "Type of lock mechanism"
                },
                "combination": {
                "type": "string",
                "description": "Combination code to unlock"
                },
                "key_required": {
                "type": "string",
                "description": "ID of the item needed to unlock"
                },
                "contains": {
                "type": "array",
                "description": "Array of item IDs contained within",
                "items": {
                    "type": "string"
                }
                }
            }
            },
            "passwordQuestion": {
            "type": "object",
            "required": ["question", "answer", "hint"],
            "properties": {
                "question": {
                "type": "string",
                "description": "The password question/prompt"
                },
                "answer": {
                "type": "string",
                "description": "The correct answer (case-insensitive)"
                },
                "hint": {
                "type": "string",
                "description": "Hint to help the player find the answer"
                }
            }
            }
        }
        }

    verified = [1, 2, 3, 5, 7]
    example_files = [f"game_jsons_and_txts/example_{v}.json" for v in verified]

    batch = BatchGameGenerator(output_dir="game_jsons_and_texts/generated/valid", n=3, schema=GAME_SCHEMA, example_files=example_files)
    batch.run()
