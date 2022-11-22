#!/bin/bash

# basic parameters
WORKSPACE="workspace"
SELFPLAY_DIR="selfplay"
LAST_STEPS_FILE="$WORKSPACE/last_steps.txt"
SETTING_FILE="torch/selfplay-setting.json"
ENGINE_NAME="Sayuri"

GAMES_PER_EPOCH=5000
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
    ENGINE_PLAY_CMD="./$ENGINE_NAME --mode selfplay --noise --random-moves-factor 0.08 --playouts 200 --komi-variance 2.5"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --selfplay-query bkp:9:7:85 --selfplay-query bkp:7:9:15"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --reduce-playouts 100 --reduce-playouts-prob 0.75"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --resign-playouts 50 --resign 0.05"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --parallel-games 4 --batch-size 2"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --cache-memory-mib 400 --early-symm-cache --first-pass-bonus"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --target-directory $SELFPLAY_DIR --weights $CURR_WEIGHTS --num-games $GAMES_PER_EPOCH"

    echo $ENGINE_PLAY_CMD
    $ENGINE_PLAY_CMD

    # step2: start training
    TRAIN_CMD="python3 torch/parser.py -j $SETTING_FILE"
    $TRAIN_CMD

    # step3: transfer the current model
    if [ -f $LAST_STEPS_FILE ]; then
        NUM_STEPS=$( cat $LAST_STEPS_FILE )
    fi
    TRANSFER_CMD="python3 torch/transfer.py -j torch/setting.json -b -n $WORKSPACE/model/s$NUM_STEPS"
    $TRANSFER_CMD
done
