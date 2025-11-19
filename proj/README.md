# Escape Room Game
`uv run python gameplay_interface.py example.json`

`--force` to force reloading game data from JSON file.

## Overview
`Prompt --> Description of game scenario --> JSON game data configuration --> TextWorld game`
This project will take a prompt from a user, passes to an LLM trained to generate a detailed 
description of an escape room game experience. The description is then converted into a JSON 
configuration file that defines the game's structure, including rooms, objects, and puzzles.
The JSON configuration is passed to a TextWorld Compiler that creates a playable text-based
escape room game utilizing the TextWorld framework and a custom gameplay interface.
The gameplay interface utilizes an LLM to both interpret player commands and generate dynamic
responses based on the game's state, the player's actions, and the experience description.

The game is played through a command-line interface where players can input commands to interact
with the game world, solve puzzles, and ultimately escape the room.
