import re
import textworld
from llm_integration import LLM_intercepter

class EscapeRoomInterface:
    def __init__(self, game_file, combination_locks=None, direction_aliases=None, password_locks=None, room_items=None):
        """
        Initialize the escape room interface
        
        Args:
            game_file: Path to the .ulx game file
            combination_locks: Dict mapping item_id to combination code
            direction_aliases: Dict mapping (room, alias) to (target, direction)
            password_locks: Dict mapping item_id to list of password questions
            room_items: Dict mapping item_id to room_id (to check location)
        """
        self.env = textworld.start(game_file)
        
        self.combination_locks = combination_locks or {}
        self.direction_aliases = direction_aliases or {}
        self.password_locks = password_locks or {}
        self.room_items = room_items or {}
        self.current_room = None
        self.unlocked_combinations = set()
        self.active_password_lock = None  # Track which password lock is being answered
        self.password_progress = {}  # Track current question index for each lock
        self.interceptor = LLM_intercepter()
        
    def reset(self):
        """Start a new game"""
        game_state = self.env.reset()
        self.unlocked_combinations = set()
        self.active_password_lock = None
        self.password_progress = {}
        
        # Initialize current room
        self._update_current_room(game_state.feedback)
        
        return game_state
    
    def _update_current_room(self, observation):
        """Update current room based on observation text"""
        obs_lower = str(observation).lower()
        
        # Try to extract room from observation
        # Look for common room identification patterns
        for room_id in self.room_items.values():
            room_name = room_id.replace('_', ' ')
            if room_name in obs_lower:
                self.current_room = room_id
                return
        
        # Fallback: check specific known rooms
        if 'security_office' in obs_lower or 'security office' in obs_lower:
            self.current_room = 'security_office'
        elif 'hallway' in obs_lower or 'corporate hallway' in obs_lower:
            self.current_room = 'hallway'
        elif 'data_vault' in obs_lower or 'data vault' in obs_lower:
            self.current_room = 'data_vault'
        elif 'spiral_staircase' in obs_lower or 'spiral staircase' in obs_lower:
            self.current_room = 'spiral_staircase'
        elif 'tower_bedroom' in obs_lower or 'tower bedroom' in obs_lower:
            self.current_room = 'tower_bedroom'
        elif 'dining_room' in obs_lower or 'dining' in obs_lower:
            self.current_room = 'dining_room'
    
    def _is_item_accessible(self, item_id):
        """Check if an item is in the current room"""
        if item_id not in self.room_items:
            return True  # If we don't know, allow it
        return self.room_items[item_id] == self.current_room

    def llm_feedback(self, feedback, user_input="", game_json=""):
        """Use an LLM to generate on-theme messages for the user"""
        return self.interceptor.llm_feedback(
            feedback=feedback,
            user_input=user_input,
            game_json=game_json
        )

    
    def step(self, command):
        """Process a command"""
        command = command.strip().lower()
        
        # Check for combination lock commands
        combo_match = re.match(r'^(?:enter|type|input|use code)\s+(\d+)(?:\s+(?:on|in|into)\s+(.+))?$', command)
        if combo_match:
            code = combo_match.group(1)
            target = combo_match.group(2)
            
            for item_id, correct_code in self.combination_locks.items():
                if code == correct_code and item_id not in self.unlocked_combinations and \
                (not target or target in item_id.replace('_', ' ') or item_id.replace('_', ' ') in target):
                    # Check if item is in current room
                    if not self._is_item_accessible(item_id):
                        class FakeState:
                            def __init__(self, msg):
                                self.feedback = msg
                                self.raw = msg
                        return (FakeState(f"You don't see that here."), 0, False)
                    
                    self.unlocked_combinations.add(item_id)
                    
                    class FakeState:
                        def __init__(self, msg):
                            self.feedback = msg
                            self.raw = msg
                    
                    return (FakeState(f"\nYou hear a satisfying *click* as the combination lock opens! The {item_id.replace('_', ' ')} is now unlocked.\n"), 0, False)
            
            class FakeState:
                def __init__(self, msg):
                    self.feedback = msg
                    self.raw = msg
            return (FakeState("The combination doesn't seem to work."), 0, False)
        
        # Handle password lock interactions
        # If actively answering questions, treat input as an answer
        if self.active_password_lock:
            item_id = self.active_password_lock
            questions = self.password_locks[item_id]
            current_q_idx = self.password_progress.get(item_id, 0)
            
            answer = command.strip().lower()
            correct_answer = questions[current_q_idx]['answer'].lower()
            
            if answer == correct_answer:
                current_q_idx += 1
                self.password_progress[item_id] = current_q_idx
                
                # Check if all questions answered
                if current_q_idx >= len(questions):
                    self.unlocked_combinations.add(item_id)
                    self.active_password_lock = None
                    
                    # Execute open door and then go through
                    item_name = item_id.replace('_', ' ')
                    
                    # First open the door
                    obs, score, done = self.env.step(f"open {item_name}")
                    
                    # Combine the messages
                    custom_message = f"Correct!\n\nThe door swings open..."
                    
                    class FakeState:
                        def __init__(self, msg):
                            self.feedback = msg
                            self.raw = msg
                    
                    return (FakeState(custom_message), score, done)
                else:
                    # Show next question
                    next_q = questions[current_q_idx]
                    
                    class FakeState:
                        def __init__(self, msg):
                            self.feedback = msg
                            self.raw = msg
                    
                    return (FakeState(f"Correct!\n{current_q_idx + 1}. {next_q['question']}\n"), 0, False)
            else:
                # Wrong answer - reset progress
                self.password_progress[item_id] = 0
                self.active_password_lock = None
                
                class FakeState:
                    def __init__(self, msg):
                        self.feedback = msg
                        self.raw = msg
                
                return (FakeState("Incorrect. Look around for clues to the answers.\n"), 0, False)
        
        # Check for open command on password-locked doors
        open_match = re.match(r'^open\s+(.+)$', command)
        if open_match:
            target = open_match.group(1)
            
            # Check combination locks
            for item_id in self.combination_locks.keys():
                item_name = item_id.replace('_', ' ')
                if target in item_name or item_name in target:
                    # Check if item is in current room
                    if not self._is_item_accessible(item_id):
                        class FakeState:
                            def __init__(self, msg):
                                self.feedback = msg
                                self.raw = msg
                        return (FakeState(f"You don't see that here."), 0, False)
                    
                    if item_id not in self.unlocked_combinations:
                        class FakeState:
                            def __init__(self, msg):
                                self.feedback = msg
                                self.raw = msg
                        return (FakeState(f"The {item_name} is locked with a combination lock. You need to enter the code first."), 0, False)
            
            # Check password locks
            for item_id, questions in self.password_locks.items():
                item_name = item_id.replace('_', ' ')
                if target in item_name or item_name in target:
                    # Check if item is in current room
                    if not self._is_item_accessible(item_id):
                        class FakeState:
                            def __init__(self, msg):
                                self.feedback = msg
                                self.raw = msg
                        return (FakeState(f"You don't see that here."), 0, False)
                    
                    if item_id not in self.unlocked_combinations:
                        # Start the password question sequence
                        self.active_password_lock = item_id
                        self.password_progress[item_id] = 0
                        
                        message = f"To open the door you must answer these questions:\n"
                        message += f"1. {questions[0]['question']}\n"
                        
                        class FakeState:
                            def __init__(self, msg):
                                self.feedback = msg
                                self.raw = msg
                        return (FakeState(message), 0, False)
        
        # Check for direction aliases
        direction_match = re.match(r'^(?:go\s+)?(\w+)$', command)
        if direction_match and self.current_room:
            direction = direction_match.group(1)
            
            if (self.current_room, direction) in self.direction_aliases:
                target_room, actual_direction = self.direction_aliases[(self.current_room, direction)]
                command = f"go {actual_direction}"
        
        # Pass command to TextWorld
        obs, score, done = self.env.step(command)
        
        # Update current room tracking
        self._update_current_room(obs.feedback)
        
        return (obs, score, done)
    
    def close(self):
        """Close the environment"""
        self.env.close()


def play_escape_room(game_file, game_json, combination_locks=None, direction_aliases=None, password_locks=None, room_items=None):
    """Interactive play session with custom command handling"""
    interface = EscapeRoomInterface(game_file, combination_locks, direction_aliases, password_locks, room_items)

    game_json_string = ""
    with open(game_json, 'r') as f:
        game_json_string = f.read()
    
    game_state = interface.reset()
    print("ORIGINAL")
    print(game_state.feedback)
    print("ENHANCED")
    print(interface.llm_feedback(game_state.feedback, game_json=game_json_string))
    
    score = 0
    moves = 0
    done = False
    
    while not done:
        try:
            command = input("\n> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Thanks for playing!")
                break
            
            obs, reward, done = interface.step(command)
            print("ORIGINAL")
            print(obs.feedback)
            print("ENHANCED")
            print(interface.llm_feedback(obs.feedback, user_input=command, game_json=game_json_string))
            
            score += reward
            moves += 1
            
            if done:
                print(f"\n{'='*60}")
                print(f"Game Over! Score: {score}, Moves: {moves}")
                print(f"{'='*60}")
                
        except KeyboardInterrupt:
            print("\n\nThanks for playing!")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    interface.close()


if __name__ == "__main__":
    import sys
    import subprocess
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python gameplay_interface.py <game_file.json>")
        print("Example: python gameplay_interface.py example.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)

    # Check for force recompile flag
    recompile = sys.argv[2].lower() == '--force' if len(sys.argv) > 2 else False

    # Create output directory
    base_name = os.path.splitext(json_file)[0].split("/")[1]
    directory_name = f"games"
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        pass
    except OSError as e:
        print(f"Error creating directory: {e}")
    
    directory_name = f"games/{base_name}"
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
        recompile = True
    except FileExistsError:
        pass
    except OSError as e:
        print(f"Error creating directory: {e}")

    # Generate output filenames
    py_file = f"games/{base_name}/game.py"
    ulx_file = f"games/{base_name}/game.ulx"
 
    
    print(f"="*60)
    print(f"Escape Room Generator & Player")
    print(f"="*60)
    print(f"Input JSON: {json_file}")
    print(f"Output Python: {py_file}")
    print(f"Output Game: {ulx_file}")
    print(f"="*60)

    if not recompile:
        print("\nStep 1: Skipped compilation...")
        print("\nStep 2: Skipped game generation...")
    else:
        # Step 1: Compile JSON to Python
        print("\nStep 1: Compiling JSON to TextWorld Python code...")
        try:
            from compiler import compile_json_to_textworld
            compile_json_to_textworld(json_file, py_file)
            print(f"✓ Generated {py_file}")
        except Exception as e:
            print(f"✗ Compilation failed: {e}")
            sys.exit(1)
        
        # Step 2: Run the Python file to generate .ulx
        print("\nStep 2: Generating .ulx game file...")
        try:
            result = subprocess.run(['python', py_file], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"✗ Game generation failed:")
                print(result.stderr)
                sys.exit(1)
            print(result.stdout)
            print(f"✓ Generated {ulx_file}")
        except Exception as e:
            print(f"✗ Failed to run {py_file}: {e}")
            sys.exit(1)
    
    # Step 3: Load game metadata
    print("\nStep 3: Loading game metadata...")
    combination_locks = {}
    direction_aliases = {}
    password_locks = {}
    room_items = {}
    
    try:
        import importlib.util
        import json
        
        spec = importlib.util.spec_from_file_location("game_module", py_file)
        game_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(game_module)
        
        if hasattr(game_module, 'COMBINATION_LOCKS'):
            combination_locks = game_module.COMBINATION_LOCKS
            print(f"✓ Loaded {len(combination_locks)} combination locks")
        
        if hasattr(game_module, 'DIRECTION_ALIASES'):
            direction_aliases = game_module.DIRECTION_ALIASES
            print(f"✓ Loaded {len(direction_aliases)} direction aliases")
        
        if hasattr(game_module, 'PASSWORD_LOCKS'):
            password_locks = game_module.PASSWORD_LOCKS
            print(f"✓ Loaded {len(password_locks)} password locks")
        
        # Load room_items mapping from JSON
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            for room in json_data['rooms']:
                room_id = room['id']
                for item in room.get('items', []):
                    if 'id' in item:
                        room_items[item['id']] = room_id
            print(f"✓ Loaded {len(room_items)} item locations")
    except Exception as e:
        print(f"⚠ Warning: Could not load metadata: {e}")
    
    # Step 4: Launch the game
    print("\n" + "="*60)
    print("Starting game...")
    print("="*60 + "\n")
    
    play_escape_room(ulx_file, json_file, combination_locks, direction_aliases, password_locks, room_items)
