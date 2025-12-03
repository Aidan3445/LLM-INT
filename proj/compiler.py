import json
from typing import Dict, List, Any
from json_validator import validate_json_file

class TextWorldCompiler:
    def __init__(self, json_data: Dict[str, Any]):
        self.data = json_data
        self.entities = {}
        self.code_lines = []
        self.object_buffer = []  # For simple objects like keys, notes
        self.container_buffer = []  # For containers and their placement
        self.aliases = {}
        self.connected_pairs = set()
        
    def compile(self) -> str:
        """Main compilation method"""
        self.add_header()
        self.create_rooms()
        self.create_items()
        self.add_locks_and_keys()
        self.add_player_start()
        self.set_quest()
        self.add_aliases()
        self.add_footer()
        
        return "\n".join(self.code_lines)
    
    def add_header(self):
        """Set the header with theme and goal and boilerplate"""
        self.code_lines.extend([
            "from textworld.generator.maker import GameMaker, get_failing_constraints",
            "from textworld.logic import State, Proposition",
            "from textworld.generator.game import Quest, Event",
            "",
            "M = GameMaker()",
            "",
            f"# Theme: {self.data['theme']}",
            f"# Goal: {self.data['goal']}",
            ""
        ])
    
    def create_rooms(self):
        """Create the rooms and connect them"""
        self.code_lines.append("# === ROOMS ===")
        
        # Create all rooms
        for room in self.data['rooms']:
            room_var = room['id']

            name = room['name'].replace("'", "\\'")
            desc = room['description'].replace("'", "\\'")

            self.code_lines.append(f"{room_var} = M.new_room('{name}')")
            self.code_lines.append(f"{room_var}.desc = '{desc}'")

            self.entities[room_var] = room_var
        
        self.code_lines.append("")
        
        # First pass: create keys
        def _get_keys(item_list):
            for item in item_list:
                if item.get('subcontainers'):
                    _get_keys(item['subcontainers'])
                if item.get('key_required'):
                    key_id = item['key_required']
                    if key_id not in self.entities:
                        self.code_lines.append(f"{key_id} = M.new(type='k', name='{key_id.replace('_', ' ')}')")
                        self.entities[key_id] = key_id

        for room in self.data['rooms']:
            _get_keys(room.get('items', []))
                    
        if any(item.get('key_required') for room in self.data['rooms'] for item in room.get('items', [])):
            self.code_lines.append("")
        
        # Connect rooms
        for room in self.data['rooms']:
            room_var = room['id']
            
            # Check for exits in items (trapdoors, doors)
            for item in room.get('items', []):
                if 'leads_to' in item:
                    target = item['leads_to']
                    connection_key = tuple(sorted([room_var, target]))
                    
                    if connection_key in self.connected_pairs:
                        continue
                    
                    self.connected_pairs.add(connection_key)
                    
                    direction = item.get('direction', 'south')
                    reverse_map = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
                    reverse = reverse_map[direction]
                    
                    self.code_lines.append(f"path_{item['id']} = M.connect({room_var}.{direction}, {target}.{reverse})")
                    
                    if item.get('locked'):
                        self.code_lines.append(f"door_{item['id']} = M.new_door(path_{item['id']}, '{item['name']}')")
                        
                        # Password locks are handled by interface, just make it closed
                        if item.get('lock_type') == 'password' or item.get('lock_type') == 'combination':
                            self.code_lines.append(f"door_{item['id']}.add_property('closed')")
                        else:
                            self.code_lines.append(f"door_{item['id']}.add_property('locked')")
                        
                        self.entities[item['id']] = f"door_{item['id']}"
                    
                    # Store aliases
                    if 'aliases' in item:
                        for alias in item['aliases']:
                            self.aliases[(room_var, alias)] = (target, direction)
        
        self.code_lines.append("")
    
    def create_items(self):
        """Create items: objects buffer, then containers buffer"""
        # Process all items and sort into buffers
        for room in self.data['rooms']:
            room_var = room['id']
            
            for item in room.get('items', []):
                if item.get('leads_to'):
                    continue  # Skip doors/paths
                
                self._process_item(item, room_var)
        
        # Write items section
        self.code_lines.append("# === ITEMS ===")
        self.code_lines.append("# Create objects (keys, notes, tools)")
        self.code_lines.extend(self.object_buffer)
        self.code_lines.append("")
        self.code_lines.append("# Create containers and place items")
        self.code_lines.extend(self.container_buffer)
        self.code_lines.append("")
    
    def _process_item(self, item: Dict, room_var: str):
        """Process an item and add to appropriate buffer"""
        item_id = item['id']
        item_name = item['name']
        
        # Skip if already processed
        if item_id in self.entities:
            return
        
        # Keys
        if item.get('key_required'):
            key_id = item['key_required']
            if key_id not in self.entities:
                self.object_buffer.append(f"{key_id} = M.new(type='k', name='{key_id.replace('_', ' ')}')")
                self.entities[key_id] = key_id
        
        # Items with subcontainers (generalized from drawers)
        if item.get('subcontainers'):
            # Parent item is just decoration
            safe_name = item_name.replace("'", "\\'")
            self.object_buffer.append(f"{item_id} = M.new(type='o', name='{safe_name}')")
            self.entities[item_id] = item_id
            self.container_buffer.append(f"{room_var}.add({item_id})")
            
            # Process each subcontainer
            for subcontainer in item['subcontainers']:
                sub_id = subcontainer['id']
                sub_name = subcontainer['name']
                
                # Create subcontainer
                safe_subname = sub_name.replace("'", "\\'")
                self.container_buffer.append(f"{sub_id} = M.new(type='c', name='{safe_subname}')")
                self.entities[sub_id] = sub_id
                
                # Set state
                if subcontainer.get('lock_type') == 'combination':
                    self.container_buffer.append(f"{sub_id}.add_property('closed')")
                elif subcontainer.get('locked'):
                    self.container_buffer.append(f"{sub_id}.add_property('locked')")
                else:
                    self.container_buffer.append(f"{sub_id}.add_property('closed')")
                
                # Place subcontainer in room
                self.container_buffer.append(f"{room_var}.add({sub_id})")
                
                # Add contents to subcontainer
                for contained_id in subcontainer.get('contains', []):
                    # First, make sure the contained item exists
                    self._ensure_item_created(contained_id)
                    self.container_buffer.append(f"{sub_id}.add({contained_id})")
            
            return
        
        # Containers (closets, chests)
        if item.get('locked') or item.get('contains'):
            safe_name = item_name.replace("'", "\\'")
            self.container_buffer.append(f"{item_id} = M.new(type='c', name='{safe_name}')")
            self.entities[item_id] = item_id
            
            # Password locks are handled by interface, so just make it closed
            if item.get('lock_type') == 'password':
                self.container_buffer.append(f"{item_id}.add_property('closed')")
            elif item.get('locked'):
                self.container_buffer.append(f"{item_id}.add_property('locked')")
            else:
                self.container_buffer.append(f"{item_id}.add_property('closed')")
            
            # Place container
            self.container_buffer.append(f"{room_var}.add({item_id})")
            
            # Add contents
            for contained_id in item.get('contains', []):
                self._ensure_item_created(contained_id)
                self.container_buffer.append(f"{item_id}.add({contained_id})")
            
            return
        
        # Regular objects (notes, tools, etc)
        safe_name = item_name.replace("'", "\\'")
        self.object_buffer.append(f"{item_id} = M.new(type='o', name='{safe_name}')")
        self.entities[item_id] = item_id
        
        # Set description for readable items
        if item.get('readable') and 'text' in item:
            text_content = item['text'].replace("'", "\\'")
            self.object_buffer.append(f"{item_id}.infos.desc = '{text_content}'")
        
        # Place in room
        self.container_buffer.append(f"{room_var}.add({item_id})")
    
    def _ensure_item_created(self, item_id: str):
        """Make sure an item is created before being referenced"""
        if item_id in self.entities:
            return
        
        # Find the item in JSON
        for room in self.data['rooms']:
            for item in room.get('items', []):
                if item['id'] == item_id:
                    # Create it
                    item_name = item['name']
                    safe_name = item_name.replace("'", "\\'")
                    self.object_buffer.append(f"{item_id} = M.new(type='o', name='{safe_name}')")
                    self.entities[item_id] = item_id
                    
                    if item.get('readable') and 'text' in item:
                        text_content = item['text'].replace("'", "\\'")
                        self.object_buffer.append(f"{item_id}.infos.desc = '{text_content}'")
                    
                    return
    
    def add_locks_and_keys(self):
        """Add facts like which keys open what doors"""
        self.code_lines.append("# === LOCKS AND KEYS ===")
        
        for room in self.data['rooms']:
            for item in room.get('items', []):
                if item.get('locked') and item.get('key_required'):
                    item_var = self.entities.get(item['id'])
                    key_var = self.entities.get(item['key_required'])
                    if item_var and key_var:
                        self.code_lines.append(f"M.add_fact('match', {key_var}, {item_var})")
                
                # Check subcontainers
                if item.get('subcontainers'):
                    for subcontainer in item['subcontainers']:
                        if subcontainer.get('locked') and subcontainer.get('key_required'):
                            sub_var = self.entities.get(subcontainer['id'])
                            key_var = self.entities.get(subcontainer['key_required'])
                            if sub_var and key_var:
                                self.code_lines.append(f"M.add_fact('match', {key_var}, {sub_var})")
        
        self.code_lines.append("")
    
    def add_player_start(self):
        """Add player start location"""
        self.code_lines.append("# === PLAYER START ===")
        starting_room = self.data.get('starting_room', self.data['rooms'][0]['id'])
        self.code_lines.append(f"M.set_player({starting_room})")
        self.code_lines.append("")
    
    def set_quest(self):
        """Set quest/destination"""
        self.code_lines.append("# === QUEST ===")
        
        # Track combination locks and password locks
        combination_locks = {}
        password_locks = {}
        
        for room in self.data['rooms']:
            # Find combination locks
            for item in room.get('items', []):
                if item.get('lock_type') == 'combination':
                    combination_locks[item['id']] = item['combination']
                
                if item.get('lock_type') == 'password':
                    password_locks[item['id']] = item.get('password_questions', [])
                
                if item.get('subcontainers'):
                    for subcontainer in item['subcontainers']:
                        if subcontainer.get('lock_type') == 'combination':
                            combination_locks[subcontainer['id']] = subcontainer['combination']
                        if subcontainer.get('lock_type') == 'password':
                            password_locks[subcontainer['id']] = subcontainer.get('password_questions', [])
            
            # Find goal room
            if room.get('is_goal'):
                self.code_lines.append(f"# Goal: Reach {room['id']}")
                self.code_lines.append(f"quest = Quest(win_events=[Event(conditions={{M.new_fact('at', M.player, {room['id']})}})])")
                self.code_lines.append("M.quests = [quest]")
        
        # Output combination locks
        if combination_locks:
            self.code_lines.append("")
            self.code_lines.append("COMBINATION_LOCKS = {")
            for item_id, combo in combination_locks.items():
                self.code_lines.append(f"    '{item_id}': '{combo}',")
            self.code_lines.append("}")
        
        # Output password locks
        if password_locks:
            self.code_lines.append("")
            self.code_lines.append("PASSWORD_LOCKS = {")
            for item_id, questions in password_locks.items():
                self.code_lines.append(f"    '{item_id}': [")
                for q in questions:
                    # Escape single quotes in questions and answers
                    question_text = q['question'].replace("'", "\\'")
                    answer_text = q['answer'].replace("'", "\\'")
                    self.code_lines.append(f"        {{'question': '{question_text}', 'answer': '{answer_text}'}},")
                self.code_lines.append(f"    ],")
            self.code_lines.append("}")
        
        self.code_lines.append("")
    
    def add_aliases(self):
        """Add direction aliases"""
        self.code_lines.append("# === DIRECTION ALIASES ===")
        self.code_lines.append("DIRECTION_ALIASES = {")
        
        for (room, alias), (target, direction) in self.aliases.items():
            self.code_lines.append(f"    ('{room}', '{alias}'): ('{target}', '{direction}'),")
        
        self.code_lines.append("}")
        self.code_lines.append("")
    
    def add_footer(self):
        """Build and compile"""
        self.code_lines.extend([
            "# === BUILD ===",
            "try:",
            "    game = M.build()",
            "except Exception as e:",
            "    bar = '~' * 60",
            "    objects = [(x.name, x.id) for x in M.findall('k')] + \\\n"
            "              [(x.name, x.id) for x in M.findall('o')] + \\\n"
            "              [(x.name, x.id) for x in M.findall('c')] + \\\n"
            "              [(x.name, x.id) for x in M.findall('r')]",
            "    fc = get_failing_constraints(M.state)",
            "    error_msg = f\"Error building the game: {e}\\n{bar}\\n{fc}\\n{bar}\\n{objects}\"",
            "    raise Exception(error_msg)",
            "",
            "game_file = M.compile('./game.ulx')",
            "print(f'Game compiled successfully to: {game_file}')",
        ])


def compile_json_to_textworld(json_file_path: str, output_file_path: str):
    """Compile JSON to TextWorld code"""
    import os
    
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
        # Validate JSON
        is_valid = validate_json_file(json_file_path)
        if not is_valid:
            sys.exit(1)
    
    compiler = TextWorldCompiler(json_data)
    code = compiler.compile()
    
    # Fix the output filename
    base_name = os.path.splitext(output_file_path)[0]
    ulx_filename = f"{base_name}.ulx"
    code = code.replace("'./game.ulx'", f"'./{ulx_filename}'")
    
    with open(output_file_path, 'w') as f:
        f.write(code)
    
    print(f"✓ Compiled {json_file_path} -> {output_file_path}")
    print(f"  Rooms: {len(json_data['rooms'])}")
    print(f"  Theme: {json_data['theme']}")
    
    return output_file_path
