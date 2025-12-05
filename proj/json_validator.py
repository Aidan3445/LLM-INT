import json
from typing import Dict, List, Any, Set, Tuple

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class TextWorldValidator:
    FORBIDDEN_NAME_CHARS = set('.,!?:;-')
    
    def __init__(self, json_data: Dict[str, Any]):
        self.data = json_data
        self.errors = []
        self.warnings = []
        self.all_item_ids = set()
        self.all_room_ids = set()
        
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validations. Returns (is_valid, errors, warnings)"""
        try:
            self._validate_top_level()
            self._collect_all_ids()
            self._validate_rooms()
            self._validate_items()
            self._validate_references()
            self._validate_lock_logic()
            
            return (len(self.errors) == 0, self.errors, self.warnings)
        except Exception as e:
            self.errors.append(f"Validation crashed: {str(e)}")
            return (False, self.errors, self.warnings)
    
    def _check_name_punctuation(self, name_value: str, name_type: str):
        """Check if a name contains forbidden punctuation characters"""
        found_chars = [c for c in name_value if c in self.FORBIDDEN_NAME_CHARS]
        if found_chars:
            self.errors.append(f"{name_type} '{name_value}' contains forbidden punctuation: {found_chars}")
    
    def _validate_top_level(self):
        """Validate top-level required fields"""
        required = ['theme', 'title', 'goal', 'rooms', 'starting_room']
        for field in required:
            if field not in self.data:
                self.errors.append(f"Missing required top-level field: '{field}'")
        
        if 'rooms' in self.data and not isinstance(self.data['rooms'], list):
            self.errors.append("'rooms' must be an array")
        
        if 'rooms' in self.data and len(self.data['rooms']) == 0:
            self.errors.append("Must have at least one room")
    
    def _collect_all_ids(self):
        """Collect all room and item IDs for reference checking"""
        for room in self.data.get('rooms', []):
            if 'id' in room:
                if room['id'] in self.all_room_ids:
                    old_id = room['id']
                    new_id = room['name'].replace(' ', '_')
                    if new_id == old_id:
                        self.errors.append(f"Duplicate room id '{old_id}' found and cannot be auto-corrected")
                    else:
                        # Update our tracking sets
                        self.all_room_ids.discard(old_id)
                        self.warnings.append(f"Duplicate room id '{old_id}' auto-corrected to '{room['id']}' to match its name")
                        room['id'] = new_id
                self.all_room_ids.add(room['id'])
        
            if 'name' in room:
                self._check_name_punctuation(room['name'], "Room name")
                
                for item in room.get('items', []):
                    self._collect_item_ids(item)
    
    def _collect_item_ids(self, item: Dict):
        """Recursively collect item IDs"""
        if 'id' in item:
            if item['id'] in self.all_item_ids:
                old_id = item['id']
                new_id = item['name'].replace(' ', '_')
                if new_id == old_id:
                    self.errors.append(f"Duplicate item id '{old_id}' found and cannot be auto-corrected")
                else:
                    # Update our tracking sets
                    self.all_item_ids.discard(old_id)
                    self.warnings.append(f"Duplicate item id '{old_id}' auto-corrected to '{item['id']}' to match its name")
                    item['id'] = new_id
            self.all_item_ids.add(item['id'])
        
        if 'name' in item:
            self._check_name_punctuation(item['name'], "Item name")
        
        # Collect from subcontainers
        for sub in item.get('subcontainers', []):
            if 'id' in sub:
                if sub['id'] in self.all_item_ids:
                    old_id = sub['id']
                    new_id = sub['name'].replace(' ', '_')
                    if new_id == old_id:
                        self.errors.append(f"Duplicate subcontainer id '{old_id}' found and cannot be auto-corrected")
                    else:
                        # Update our tracking sets
                        self.all_item_ids.discard(old_id)
                        self.warnings.append(f"Duplicate subcontainer id '{old_id}' auto-corrected to '{sub['id']}' to match its name")
                        sub['id'] = new_id
                self.all_item_ids.add(sub['id'])
            
            if 'name' in sub:
                self._check_name_punctuation(sub['name'], "Subcontainer name")
    
    def _validate_rooms(self):
        """Validate room structure"""
        goal_rooms = []
        
        for room in self.data.get('rooms', []):
            # Required fields
            for field in ['id', 'name', 'description']:
                if field not in room:
                    self.errors.append(f"Room missing required field: '{field}'")
            
            # Check goal room
            if room.get('is_goal'):
                goal_rooms.append(room['id'])
                if 'win_message' not in room:
                    self.errors.append(f"Goal room '{room['id']}' missing 'win_message'")
        
        # Exactly one goal room
        if len(goal_rooms) == 0:
            self.errors.append("Must have exactly one room with 'is_goal': true")
        elif len(goal_rooms) > 1:
            self.errors.append(f"Multiple goal rooms found: {goal_rooms}. Only one allowed.")
        
        # Starting room exists
        if 'starting_room' in self.data:
            if self.data['starting_room'] not in self.all_room_ids:
                self.errors.append(f"starting_room '{self.data['starting_room']}' does not exist")
    
    def _validate_items(self):
        """Validate all items in all rooms"""
        for room in self.data.get('rooms', []):
            for item in room.get('items', []):
                self._validate_item(item, room['id'])
    
    def _validate_item(self, item: Dict, room_id: str):
        """Validate a single item"""
        # Required fields
        for field in ['id', 'name', 'description']:
            if field not in item:
                self.errors.append(f"Item in '{room_id}' missing required field: '{field}'")
                return
        
        item_id = item['id']
        
        # Determine item type
        is_door = 'leads_to' in item
        has_subcontainers = 'subcontainers' in item
        is_container = 'contains' in item or 'locked' in item or 'searchable' in item
        is_readable = item.get('readable', False)
        
        # DOOR validations
        if is_door:
            if 'direction' not in item:
                self.errors.append(f"Door '{item_id}' missing 'direction'")
            elif item['direction'] not in ['north', 'south', 'east', 'west']:
                self.errors.append(f"Door '{item_id}' has invalid direction: '{item['direction']}'")
            
            if item['leads_to'] not in self.all_room_ids:
                self.errors.append(f"Door '{item_id}' leads_to non-existent room: '{item['leads_to']}'")
            
            # Doors can't have contains or subcontainers
            if 'contains' in item:
                self.errors.append(f"Door '{item_id}' cannot have 'contains'")
            if has_subcontainers:
                self.errors.append(f"Door '{item_id}' cannot have 'subcontainers'")

            # Door id and name must be the same with `_` <=> ` `
            # Auto-fix if mismatched
            expected_id = item['name'].replace(' ', '_')
            if item_id != expected_id:
                old_id = item['id']
                item['id'] = expected_id
                # Update our tracking sets
                self.all_item_ids.discard(old_id)
                self.all_item_ids.add(expected_id)
                self.warnings.append(f"Door id '{old_id}' auto-corrected to '{expected_id}' to match its name")
        
        # SUBCONTAINER validations
        if has_subcontainers:
            if is_door:
                self.errors.append(f"Item '{item_id}' cannot be both door and have subcontainers")
            
            if not isinstance(item['subcontainers'], list):
                self.errors.append(f"Item '{item_id}' subcontainers must be an array")
            else:
                for sub in item['subcontainers']:
                    self._validate_subcontainer(sub, item_id)
        
        # CONTAINER validations
        if is_container and not is_door and not has_subcontainers:
            if 'contains' in item and not isinstance(item['contains'], list):
                self.errors.append(f"Container '{item_id}' contains must be an array")
        
        # READABLE validations
        if is_readable:
            if 'text' not in item:
                self.errors.append(f"Readable item '{item_id}' missing 'text' field")
        
        # LOCK validations
        self._validate_locks(item, item_id, is_door)
    
    def _validate_subcontainer(self, sub: Dict, parent_id: str):
        """Validate a subcontainer"""
        if 'id' not in sub:
            self.errors.append(f"Subcontainer in '{parent_id}' missing 'id'")
            return
        
        if 'name' not in sub:
            self.errors.append(f"Subcontainer '{sub['id']}' missing 'name'")
        
        if 'contains' in sub and not isinstance(sub['contains'], list):
            self.errors.append(f"Subcontainer '{sub['id']}' contains must be an array")
        
        # Subcontainers can have locks
        self._validate_locks(sub, sub['id'], is_door=False)
    
    def _validate_locks(self, item: Dict, item_id: str, is_door: bool):
        """Validate locking mechanisms"""
        has_key = 'key_required' in item
        has_lock_type = 'lock_type' in item
        is_locked = item.get('locked', False)
        
        # Can't have both key and lock_type
        if has_key and has_lock_type:
            self.errors.append(f"Item '{item_id}' cannot have both 'key_required' and 'lock_type'")
        
        # If locked, must have a lock mechanism
        if is_locked and not has_key and not has_lock_type:
            self.errors.append(f"Item '{item_id}' is locked but has no key_required or lock_type")
        
        # Validate lock_type
        if has_lock_type:
            lock_type = item['lock_type']
            
            if lock_type not in ['combination', 'password']:
                self.errors.append(f"Item '{item_id}' has invalid lock_type: '{lock_type}'")
            
            if lock_type == 'combination':
                if 'combination' not in item:
                    self.errors.append(f"Item '{item_id}' has lock_type 'combination' but missing 'combination' field")
                elif not isinstance(item['combination'], str):
                    self.errors.append(f"Item '{item_id}' combination must be a string")
            
            if lock_type == 'password':
                if 'password_questions' not in item:
                    self.errors.append(f"Item '{item_id}' has lock_type 'password' but missing 'password_questions'")
                elif not isinstance(item['password_questions'], list):
                    self.errors.append(f"Item '{item_id}' password_questions must be an array")
                elif len(item['password_questions']) == 0:
                    self.errors.append(f"Item '{item_id}' password_questions cannot be empty")
                else:
                    for i, q in enumerate(item['password_questions']):
                        if 'question' not in q:
                            self.errors.append(f"Item '{item_id}' password question {i} missing 'question'")
                        if 'answer' not in q:
                            self.errors.append(f"Item '{item_id}' password question {i} missing 'answer'")
                
                # Password locks only valid for doors
                if not is_door:
                    self.warnings.append(f"Item '{item_id}' uses password lock but is not a door (may not work as expected)")
    
    def _validate_references(self):
        """Validate all ID references"""
        for room in self.data.get('rooms', []):
            for item in room.get('items', []):
                self._validate_item_references(item)
    
    def _validate_item_references(self, item: Dict):
        """Validate references in an item"""
        item_id = item.get('id', 'unknown')
        
        # Check key_required
        if 'key_required' in item:
            # Keys can be auto-generated, so just warn if not found
            if item['key_required'] not in self.all_item_ids:
                self.warnings.append(f"Item '{item_id}' references key '{item['key_required']}' which is not explicitly defined (will be auto-generated)")
        
        # Check contains
        for contained_id in item.get('contains', []):
            if contained_id not in self.all_item_ids:
                self.errors.append(f"Item '{item_id}' contains non-existent item: '{contained_id}'")
        
        # Check subcontainers
        for sub in item.get('subcontainers', []):
            if 'key_required' in sub:
                if sub['key_required'] not in self.all_item_ids:
                    self.warnings.append(f"Subcontainer '{sub.get('id', 'unknown')}' references key '{sub['key_required']}' which is not defined (will be auto-generated)")
            
            for contained_id in sub.get('contains', []):
                if contained_id not in self.all_item_ids:
                    self.errors.append(f"Subcontainer '{sub.get('id', 'unknown')}' contains non-existent item: '{contained_id}'")
    
    def _validate_lock_logic(self):
        """Validate game logic around locks and accessibility"""
        # Check if goal room is reachable
        # This is a simple check - doesn't handle complex key chains
        
        # Check for orphaned keys (keys that don't unlock anything)
        referenced_keys = set()
        for room in self.data.get('rooms', []):
            for item in room.get('items', []):
                if 'key_required' in item:
                    referenced_keys.add(item['key_required'])
                for sub in item.get('subcontainers', []):
                    if 'key_required' in sub:
                        referenced_keys.add(sub['key_required'])
        
        # Find items that look like keys but aren't referenced
        for item_id in self.all_item_ids:
            if 'key' in item_id.lower() and item_id not in referenced_keys:
                self.warnings.append(f"Item '{item_id}' looks like a key but isn't used to unlock anything")


def validate_json_file(json_file_path: str) -> bool:
    """Validate a JSON file and print results"""
    try:
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON: {e}")
    except FileNotFoundError:
        raise Exception(f"File not found: {json_file_path}")
    
    validator = TextWorldValidator(json_data)
    is_valid, errors, warnings = validator.validate()
    
    print(f"\n{'='*60}")
    print(f"Validation Results: {json_file_path}")
    print(f"{'='*60}")

    feedback = ""
    if errors:
        all_errors = "\n  • ".join(errors)
        feedback += f"\n❗ ERRORS ({len(errors)}):\n  • {all_errors}"
    if warnings:
        feedback += f"\n⚠️  WARNINGS ({len(warnings)}):"
        for warning in warnings:
            feedback += f"\n  • {warning}"

    if is_valid:
        print(f"\n✅ Validation passed!")
        print(f"   Rooms: {len(validator.all_room_ids)}")
        print(f"   Items: {len(validator.all_item_ids)}")
        if warnings:
            print(feedback)
            print(f"\n⚠️ Updating file with auto-corrections...")
            with open(json_file_path, 'w') as f:
                json.dump(json_data, f, indent=4)
    else:
        feedback += "\n❗ Validation failed."
        raise ValidationError(feedback)
    
    return is_valid
