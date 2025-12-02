import os
import json
import dspy
import random


# Need to configure dspy model

class GenerateGameJSON(dspy.Signature):
    theme = dspy.InputField()
    title = dspy.InputField()
    goal = dspy.InputField()
    output_json = dspy.OutputField(desc="A VALID JSON dictionary matching the required schema exactly.")
    
class GameGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateGameJSON)

    def forward(self, theme, title, goal):
        result = self.generate(theme=theme, title=title, goal=goal)
        text = result.output_json.strip()

        try:
            data = json.loads(text)
            return data
        except Exception:
            raise ValueError(f"Invalid JSON received:\n{text}")

class BatchGameGenerator:
    def __init__(self, output_dir="game_jsons_and_txts", n=300):
        self.output_dir = output_dir
        self.n = n
        os.makedirs(output_dir, exist_ok=True)
        self.generator = GameGenerator()

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

            try:
                data = self.generator(theme=theme, title=title, goal=goal)
                path = os.path.join(self.output_dir, f"example_{i+1}.json")

                with open(path, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"✓ Saved {path}")

            except Exception as e:
                print(f"✗ Failed generating #{i+1}: {e}")
                
                
if __name__ == "__main__":
    batch = BatchGameGenerator(output_dir="games_jsons_and_txts", n=300)
    batch.run()