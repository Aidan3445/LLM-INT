import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import json

    # file called example.json
    with open('example.json', 'r') as file:
        example_obj = json.load(file)

    example_obj
    return (json,)


@app.cell
def _():
    from typing import Dict, List, Any

    class TextWorldCompiler:
        def __init__(self, json_data: Dict[str, Any]):
            self.data = json_data
            self.entities = {}  # Track all created entities
            self.code_lines = []
            self.aliases = {}  # Track direction aliases for gameplay interface
            self.connected_pairs = set()  # Track which room pairs are already connected
        
        def compile(self) -> str:
            """Main compilation method"""
            self.add_header()
            self.create_rooms()
            self.create_items()
            self.setup_containers_and_locks()
            self.setup_puzzles()
            self.set_starting_conditions()
            self.add_aliases_data()
            self.add_footer()
        
            return "\n".join(self.code_lines)
    
        def add_header(self):
            """Add imports and game initialization"""
            self.code_lines.extend([
                "from textworld import GameMaker",
                "from textworld.logic import State, Proposition",
                "from textworld.generator.game import Quest, Event",
                "",
                "# Create the game maker",
                "M = GameMaker()",
                "",
                f"# Theme: {self.data['theme']}",
                f"# Goal: {self.data['goal']}",
                ""
            ])
    
        def create_rooms(self):
            """Create all rooms"""
            self.code_lines.append("# === ROOMS ===")
        
            for room in self.data['rooms']:
                room_var = room['id']
                room_name = room['name']
                room_desc = room['description']
            
                self.code_lines.append(f"{room_var} = M.new_room('{room_name}')")
                self.code_lines.append(f"{room_var}.desc = '{room_desc}'")
                self.entities[room['id']] = room_var
                self.code_lines.append("")
        
            self.code_lines.append("")
        
            # Connect rooms via exits
            self.code_lines.append("# Connect rooms")
        
            # Available directions in TextWorld (north, south, east, west)
            for room in self.data['rooms']:
                room_var = room['id']
            
                # Handle trapdoors and special exits in items
                for item in room.get('items', []):
                    if 'leads_to' in item:
                        target_room = item['leads_to']
                    
                        # Create a unique key for this connection (order-independent)
                        connection_key = tuple(sorted([room_var, target_room]))
                    
                        # Skip if we've already connected these rooms
                        if connection_key in self.connected_pairs:
                            self.code_lines.append(f"# Skipping duplicate connection: {room_var} <-> {target_room}")
                            continue
                    
                        self.connected_pairs.add(connection_key)
                    
                        # Get the actual direction to use (defaults to south)
                        actual_direction = item.get('direction', 'south')
                        reverse_map = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
                        reverse_direction = reverse_map.get(actual_direction, 'north')
                    
                        self.code_lines.append(f"# Connecting via {item['name']} (actual direction: {actual_direction})")
                    
                        # Store aliases for this exit (forward direction only)
                        if 'aliases' in item:
                            for alias in item['aliases']:
                                # Map: (from_room, alias) -> (to_room, actual_direction)
                                self.aliases[(room_var, alias)] = (target_room, actual_direction)
                            self.code_lines.append(f"# Aliases from {room_var}: {', '.join(item['aliases'])}")
                    
                        self.code_lines.append(f"path_{item['id']} = M.connect({room_var}.{actual_direction}, {target_room}.{reverse_direction})")
                    
                        if item.get('locked'):
                            self.code_lines.append(f"door_{item['id']} = M.new_door(path_{item['id']}, '{item['name']}')")
                            self.code_lines.append(f"door_{item['id']}.add_property('locked')")
                            self.entities[item['id']] = f"door_{item['id']}"
            
                # Handle normal exits (rarely used since connections are bidirectional)
                for exit_info in room.get('exits', []):
                    if exit_info.get('leads_to'):
                        target = exit_info['leads_to']
                    
                        # Create a unique key for this connection (order-independent)
                        connection_key = tuple(sorted([room_var, target]))
                    
                        # Check if connection already exists
                        if connection_key in self.connected_pairs:
                            # Connection exists, just add aliases for this direction
                            if 'aliases' in exit_info:
                                direction = exit_info.get('direction', 'north')
                                for alias in exit_info['aliases']:
                                    self.aliases[(room_var, alias)] = (target, direction)
                                self.code_lines.append(f"# Adding aliases from {room_var}: {', '.join(exit_info['aliases'])}")
                            continue
                    
                        self.connected_pairs.add(connection_key)
                    
                        direction = exit_info.get('direction', 'north')
                        reverse_dir = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
                        reverse = reverse_dir.get(direction, 'south')
                    
                        # Store aliases for this exit
                        if 'aliases' in exit_info:
                            for alias in exit_info['aliases']:
                                self.aliases[(room_var, alias)] = (target, direction)
                            self.code_lines.append(f"# Exit {direction} from {room_var} with aliases: {', '.join(exit_info['aliases'])}")
                    
                        self.code_lines.append(f"M.connect({room_var}.{direction}, {target}.{reverse})")
        
            self.code_lines.append("")
    
        def create_items(self):
            """Create all items and objects"""
            self.code_lines.append("# === ITEMS ===")
        
            for room in self.data['rooms']:
                room_var = room['id']
            
                for item in room.get('items', []):
                    self._create_item(item, room_var)
        
            self.code_lines.append("")
    
        def _create_item(self, item: Dict, parent_location: str, indent: int = 0):
            """Recursively create an item and its contents"""
            item_id = item['id']
            item_name = item['name']
        
            # Skip items that are actually doors/paths
            if 'leads_to' in item:
                return
        
            # Determine item type
            if item.get('locked') or item.get('contains'):
                # Container
                item_var = f"{item_id}"
                self.code_lines.append(f"{item_var} = M.new(type='c', name='{item_name}')")
                self.entities[item_id] = item_var
            
                # Containers must have a state: open, closed, or locked
                if item.get('locked'):
                    self.code_lines.append(f"{item_var}.add_property('locked')")
                else:
                    # Default to closed if it's a container but not locked
                    self.code_lines.append(f"{item_var}.add_property('closed')")
        
            elif item.get('readable'):
                # Readable item (book, note, etc)
                item_var = f"{item_id}"
                self.code_lines.append(f"{item_var} = M.new(type='o', name='{item_name}')")
                self.entities[item_id] = item_var
        
            elif item.get('drawers'):
                # Special handling for dresser with drawers - make it a container
                item_var = f"{item_id}"
                self.code_lines.append(f"{item_var} = M.new(type='c', name='{item_name}')")
                self.code_lines.append(f"{item_var}.add_property('open')  # Dresser is open so you can access drawers")
                self.entities[item_id] = item_var
            
                # Create each drawer as a separate container inside the dresser
                for drawer in item['drawers']:
                    drawer_var = drawer['id']
                    drawer_name = drawer['name']
                    self.code_lines.append(f"{drawer_var} = M.new(type='c', name='{drawer_name}')")
                    self.entities[drawer['id']] = drawer_var
                
                    # Set drawer state
                    if drawer.get('locked'):
                        self.code_lines.append(f"{drawer_var}.add_property('locked')")
                    else:
                        self.code_lines.append(f"{drawer_var}.add_property('closed')")
                
                    # Place drawer IN the dresser (container within container)
                    self.code_lines.append(f"{item_var}.add({drawer_var})")
                
                    # Create and add contents to drawer
                    for contained_id in drawer.get('contains', []):
                        # Find the item definition in our JSON
                        contained_item = None
                        for room in self.data['rooms']:
                            for room_item in room.get('items', []):
                                if room_item['id'] == contained_id:
                                    contained_item = room_item
                                    break
                    
                        if contained_item:
                            # Create the item if it doesn't exist yet
                            if contained_id not in self.entities:
                                contained_var = f"{contained_id}"
                                self.code_lines.append(f"{contained_var} = M.new(type='o', name='{contained_item['name']}')")
                                self.entities[contained_id] = contained_var
                            # Add to drawer
                            self.code_lines.append(f"{drawer_var}.add({self.entities[contained_id]})")
                
                    self.code_lines.append("")
            
                # Now place the dresser in the room
                self.code_lines.append(f"{parent_location}.add({item_var})")
                self.code_lines.append("")
                return  # Early return since we handled placement
        
            else:
                # Regular object
                item_var = f"{item_id}"
                self.code_lines.append(f"{item_var} = M.new(type='o', name='{item_name}')")
                self.entities[item_id] = item_var
        
            # Place item in location
            self.code_lines.append(f"{parent_location}.add({item_var})")
            self.code_lines.append("")
        
            # Handle nested items (contents)
            for contained in item.get('contains', []):
                if isinstance(contained, str):
                    # Just an ID reference, will be created elsewhere
                    pass
                elif isinstance(contained, dict):
                    self._create_item(contained, item_var)
    
        def setup_containers_and_locks(self):
            """Set up container relationships and locks"""
            self.code_lines.append("# === LOCKS AND KEYS ===")
        
            for room in self.data['rooms']:
                for item in room.get('items', []):
                    self._setup_locks(item)
                
                    # Handle drawers
                    if 'drawers' in item:
                        for drawer in item['drawers']:
                            self._setup_locks(drawer)
        
            self.code_lines.append("")
    
        def _setup_locks(self, item: Dict):
            """Setup locks for a specific item"""
            if item.get('locked') and item.get('key_required'):
                item_var = self.entities.get(item['id'])
                key_var = self.entities.get(item['key_required'])
            
                if item_var and key_var:
                    self.code_lines.append(f"# {item['name']} locked by {item['key_required']}")
                    self.code_lines.append(f"M.add_fact('match', {key_var}, {item_var})")
    
        def setup_puzzles(self):
            """Setup special puzzles like combinations and riddles"""
            self.code_lines.append("# === PUZZLES ===")
        
            for room in self.data['rooms']:
                for item in room.get('items', []):
                    # Combination locks
                    if item.get('lock_type') == 'combination':
                        combination = item['combination']
                        item_var = self.entities.get(item['id'])
                        self.code_lines.append(f"# {item['name']} has combination: {combination}")
                        self.code_lines.append(f"# Player must type: 'enter {combination} on {item['name']}'")
                        self.code_lines.append("")
                
                    # Riddle locks
                    if item.get('lock_type') == 'riddle':
                        item_var = self.entities.get(item['id'])
                        self.code_lines.append(f"# {item['name']} has riddle puzzle")
                        for q in item.get('riddle_questions', []):
                            self.code_lines.append(f"# Q: {q['question']} -> A: {q['answer']}")
                        self.code_lines.append("")
                
                    # Handle drawers with combinations
                    if 'drawers' in item:
                        for drawer in item['drawers']:
                            if drawer.get('lock_type') == 'combination':
                                combination = drawer['combination']
                                self.code_lines.append(f"# {drawer['name']} has combination: {combination}")
                                self.code_lines.append("")
        
            self.code_lines.append("")
    
        def set_starting_conditions(self):
            """Set player starting position and win conditions"""
            self.code_lines.append("# === GAME SETUP ===")
        
            starting_room = self.data.get('starting_room', self.data['rooms'][0]['id'])
            self.code_lines.append(f"M.set_player({starting_room})")
            self.code_lines.append("")
        
            # Create a proper quest using Quest and Event objects
            goal_room = None
            for room in self.data['rooms']:
                if room.get('is_goal'):
                    goal_room = room['id']
                    win_message = room.get('win_message', 'You win!')
                    self.code_lines.append(f"# Win condition: Reach {goal_room}")
                    self.code_lines.append(f"# Win message: {win_message}")
                    self.code_lines.append("")
                
                    # Create a quest that triggers when player reaches the goal room
                    self.code_lines.append("# Create quest: reach the dining room")
                    self.code_lines.append(f"quest = Quest(win_events=[")
                    self.code_lines.append(f"    Event(conditions={{M.new_fact('at', M.player, {goal_room})}})")
                    self.code_lines.append(f"])")
                    self.code_lines.append("M.quests = [quest]")
        
            self.code_lines.append("")
    
        def add_aliases_data(self):
            """Add alias mapping as Python data structure for gameplay interface"""
            self.code_lines.append("# === DIRECTION ALIASES ===")
            self.code_lines.append("# Map of (room, alias) -> (target_room, actual_direction)")
            self.code_lines.append("# Use this in your gameplay interface to translate player commands")
            self.code_lines.append("DIRECTION_ALIASES = {")
        
            for (room, alias), (target, direction) in self.aliases.items():
                self.code_lines.append(f"    ('{room}', '{alias}'): ('{target}', '{direction}'),")
        
            self.code_lines.append("}")
            self.code_lines.append("")
    
        def add_footer(self):
            """Add game generation code"""
            self.code_lines.extend([
                "# === GENERATE GAME ===",
                "game = M.build()",
                "",
                "# Compile to .ulx file",
                "game_file = M.compile('./castle_escape_game.ulx')",
                "print(f'Game compiled successfully to: {game_file}')",
                ""
            ])
    return (TextWorldCompiler,)


@app.cell
def _(TextWorldCompiler, json):
    def compile_json_to_textworld(json_file_path: str, output_file_path: str):
        """
        Main function to compile JSON to TextWorld code
    
        Args:
            json_file_path: Path to input JSON file
            output_file_path: Path to output Python file
        """
        # Load JSON
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
    
        # Compile
        compiler = TextWorldCompiler(json_data)
        textworld_code = compiler.compile()
    
        # Write output
        with open(output_file_path, 'w') as f:
            f.write(textworld_code)
    
        print(f"✓ Compiled {json_file_path} -> {output_file_path}")
        print(f"  Rooms: {len(json_data['rooms'])}")
        print(f"  Theme: {json_data['theme']}")

        return textworld_code

    print(compile_json_to_textworld("example.json", "castle_escape_game.py"))
    return


if __name__ == "__main__":
    app.run()
