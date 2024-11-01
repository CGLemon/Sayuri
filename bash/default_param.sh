#!/bin/bash

# Training setting file path.
SETTING_FILE="configs/selfplay-setting.json"

# Save the current training status here.
WORKSPACE="workspace"

# Self-play engine picks the last weights here.
WEIGHTS_DIR="weights"

# Self-play engine saves self-play data here.
SELFPLAY_DIR="selfplay"

# Self-paly setting file path.
CONFIG_FILE="configs/selfplay-config.txt"

# Kill file path.
KILL_FILE="kill.txt"

# Execute the infinite loop if it is true.
EXECUTE_LOOP=1
