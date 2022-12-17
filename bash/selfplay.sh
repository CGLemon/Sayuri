#!/bin/bash

# directory parameters
WORKSPACE="workspace"
SELFPLAY_DIR="selfplay"
LAST_STEPS_FILE="$WORKSPACE/last_steps.txt"
SETTING_FILE="selfplay-setting.json"
CONFIG_FILE="selfplay-config.txt"
ENGINE_NAME="Sayuri"

# games parameters
GAMES_PER_EPOCH=10000
MAX_TRAINING_EPOCHES=20

safe_mkdir()
{
    if [ ! -d "$1" ]; then
        mkdir $1
    fi
}

safe_mkdir $SELFPLAY_DIR
safe_mkdir $WORKSPACE

for ((i=0; i<$MAX_TRAINING_EPOCHES; i++)); do
    NUM_STEPS=0
    if [ -f $LAST_STEPS_FILE ]; then
        NUM_STEPS=$( cat $LAST_STEPS_FILE )
    fi
    CURR_WEIGHTS="$WORKSPACE/model/s$NUM_STEPS.bin.txt"

    # step1: self-play
    ENGINE_PLAY_CMD="./$ENGINE_NAME --mode selfplay --config $CONFIG_FILE"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --target-directory $SELFPLAY_DIR"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --weights $CURR_WEIGHTS"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --num-games $GAMES_PER_EPOCH"

    echo $ENGINE_PLAY_CMD
    $ENGINE_PLAY_CMD

    # step2: start training
    TRAIN_CMD="python3 torch/parser.py -j $SETTING_FILE"
    $TRAIN_CMD

    # step3: transfer the current model
    if [ -f $LAST_STEPS_FILE ]; then
        NUM_STEPS=$( cat $LAST_STEPS_FILE )
    fi
    TRANSFER_CMD="python3 torch/transfer.py -j $SETTING_FILE -b -n $WORKSPACE/model/s$NUM_STEPS"
    $TRANSFER_CMD
done
