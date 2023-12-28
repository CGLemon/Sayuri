#!/bin/bash

# directory parameters
WORKSPACE="workspace"
SELFPLAY_DIR="selfplay"
WEIGHTS_DIR="$WORKSPACE/weights"
LAST_STEPS_FILE="$WORKSPACE/last_steps.txt"
SETTING_FILE="selfplay-setting.json"
KILL_FILE="kill.txt"
CONFIG_FILE="selfplay-config.txt"
ENGINE_NAME="sayuri"

safe_mkdir()
{
    if [ ! -d "$1" ]; then
        mkdir $1
    fi
}

safe_mkdir $SELFPLAY_DIR
safe_mkdir $WORKSPACE
safe_mkdir $WEIGHTS_DIR

while true
do
    # Do the self-play.
    ENGINE_PLAY_CMD="./$ENGINE_NAME --mode selfplay --config $CONFIG_FILE"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --target-directory $SELFPLAY_DIR"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --weights-dir $WEIGHTS_DIR"

    echo $ENGINE_PLAY_CMD
    $ENGINE_PLAY_CMD

    # Train a new model.
    TRAIN_CMD="python3 torch/main.py -j $SETTING_FILE"
    $TRAIN_CMD

    # Stop the loop if we find the kill file.
    if [ -f $KILL_FILE ]; then 
        echo "Find the kill file. Stop the self-play loop."
        rm $KILL_FILE
        break
    fi
done
