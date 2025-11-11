import re
import textworld

class EscapeRoomInterface:
    def __init__(self, game_file, combination_locks=None, direction_aliases=None):
        """
        Initialize the escape room interface
        
        Args:
            game_file: Path to the .ulx game file
            combination_locks: Dict mapping item_id to combination code
            direction_aliases: Dict mapping (room, alias) to (target, direction)
        """
        # Use the basic textworld.start API
        self.env = textworld.start(game_file)
        
        self.combination_locks = combination_locks or {}
        self.direction_aliases = direction_aliases or {}
        self.current_room = None
        self.unlocked_combinations = set()
        
    def reset(self):
        """Start a new game"""
        game_state = self.env.reset()
        self.unlocked_combinations = set()
        return game_state
    
    def step(self, command):
        """
        Process a command, handling custom logic before passing to TextWorld
        
        Args:
            command: Player's text command
            
        Returns:
            (observation, reward, done)
        """
        command = command.strip().lower()
        
        # Check for combination lock commands
        # Patterns: "enter 476", "enter 476 on middle drawer", "type 476"
        combo_match = re.match(r'^(?:enter|type|input|use code)\s+(\d+)(?:\s+(?:on|in|into)\s+(.+))?$', command)
        if combo_match:
            code = combo_match.group(1)
            target = combo_match.group(2)
            
            # Find which lock this code opens
            for item_id, correct_code in self.combination_locks.items():
                if code == correct_code and item_id not in self.unlocked_combinations:
                    # Check if target specified and matches
                    if target:
                        # Normalize the target name (remove "the", extra spaces)
                        target_clean = target.replace('the ', '').strip()
                        item_name_clean = item_id.replace('_', ' ')
                        
                        if target_clean not in item_name_clean and item_name_clean not in target_clean:
                            continue  # Wrong target specified
                    
                    # Mark as unlocked in our tracker
                    self.unlocked_combinations.add(item_id)
                    
                    # Create a fake game state with success message
                    class FakeState:
                        def __init__(self, msg):
                            self.feedback = msg
                            self.raw = msg
                    
                    custom_message = f"\nYou hear a satisfying *click* as the combination lock opens! The {item_id.replace('_', ' ')} is now unlocked.\n"
                    return (FakeState(custom_message), 0, False)
            
            # Wrong code or already unlocked
            class FakeState:
                def __init__(self, msg):
                    self.feedback = msg
                    self.raw = msg
            return (FakeState("The combination doesn't seem to work."), 0, False)
        
        # Intercept open commands for combination-locked items
        open_match = re.match(r'^open\s+(.+)$', command)
        if open_match:
            target = open_match.group(1)
            
            # Check if trying to open a combination-locked item
            for item_id in self.combination_locks.keys():
                item_name = item_id.replace('_', ' ')
                if target in item_name or item_name in target:
                    if item_id not in self.unlocked_combinations:
                        class FakeState:
                            def __init__(self, msg):
                                self.feedback = msg
                                self.raw = msg
                        return (FakeState(f"The {item_name} is locked with a combination lock. You need to enter the code first."), 0, False)
        
        # Check for direction aliases (e.g., "go down" -> "go south")
        # Pattern: "go <alias>" or just "<alias>"
        direction_match = re.match(r'^(?:go\s+)?(\w+)$', command)
        if direction_match and self.current_room:
            direction = direction_match.group(1)
            
            # Check if this is an alias
            if (self.current_room, direction) in self.direction_aliases:
                target_room, actual_direction = self.direction_aliases[(self.current_room, direction)]
                command = f"go {actual_direction}"
                print(f"[Translating '{direction}' to '{actual_direction}']")
        
        # Pass command to TextWorld
        obs, score, done = self.env.step(command)
        
        # Update current room tracking (simple heuristic)
        obs_lower = str(obs).lower()
        if 'tower_bedroom' in obs_lower or 'tower bedroom' in obs_lower:
            self.current_room = 'tower_bedroom'
        elif 'spiral_staircase' in obs_lower or 'spiral staircase' in obs_lower:
            self.current_room = 'spiral_staircase'
        elif 'dining_room' in obs_lower or 'dining' in obs_lower:
            self.current_room = 'dining_room'
        
        return (obs, score, done)
    
    def close(self):
        """Close the environment"""
        self.env.close()


def play_escape_room(game_file, combination_locks=None, direction_aliases=None):
    """
    Interactive play session with custom command handling
    
    Args:
        game_file: Path to the .ulx game file
        combination_locks: Dict mapping item_id to combination code
        direction_aliases: Dict mapping (room, alias) to (target, direction)
    """
    interface = EscapeRoomInterface(game_file, combination_locks, direction_aliases)
    
    # Start game
    game_state = interface.reset()
    print(game_state.feedback)
    
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
            print(obs.feedback)
            
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
        print("Example: python gameplay_interface.py castle_escape.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)
    
    # Generate output filenames based on JSON filename
    base_name = os.path.splitext(json_file)[0]
    py_file = f"{base_name}_game.py"
    ulx_file = f"{base_name}_game.ulx"
    
    print(f"="*60)
    print(f"Escape Room Generator & Player")
    print(f"="*60)
    print(f"Input JSON: {json_file}")
    print(f"Output Python: {py_file}")
    print(f"Output Game: {ulx_file}")
    print(f"="*60)
    
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
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("game_module", py_file)
        game_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(game_module)
        
        if hasattr(game_module, 'COMBINATION_LOCKS'):
            combination_locks = game_module.COMBINATION_LOCKS
            print(f"✓ Loaded {len(combination_locks)} combination locks")
        
        if hasattr(game_module, 'DIRECTION_ALIASES'):
            direction_aliases = game_module.DIRECTION_ALIASES
            print(f"✓ Loaded {len(direction_aliases)} direction aliases")
    except Exception as e:
        print(f"⚠ Warning: Could not load metadata: {e}")
    
    # Step 4: Launch the game
    print("\n" + "="*60)
    print("Starting game...")
    print("="*60 + "\n")
    
    play_escape_room(ulx_file, combination_locks, direction_aliases)
