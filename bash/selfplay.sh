#!/bin/bash

usage()
{
    echo "usage: -e <int> | init epoches"
}

safe_mkdir()
{
    if [ ! -d "$1" ]; then
        mkdir $1
        echo "mkdir $1"
    fi
}

safe_rmdir()
{
    if [ -d "$1" ]; then
        rm -r $1
        echo "rm -r $1"
    fi
}

if [ "$1" = "-h" ] || [ $# = 0 ]; then
    usage
    exit 1
fi

while getopts e: flag
do
    case "${flag}" in
        e) init_epoches=${OPTARG};;
    esac
done

# basic parameters
WORKSPACE="workspace"
TRAINING_SET_DIR="train-dir"

GAMES_PER_EPOCH=1000
STEPS_PER_EPOCH=1000

TRAINING_EPOCHES=10
DELAY_BUF_EPOCHES=10

ENGINE_NAME="Sayuri"
INIT_EPOCHES=$init_epoches

safe_mkdir $TRAINING_SET_DIR
safe_mkdir $WORKSPACE
safe_mkdir "$WORKSPACE/model"

for ((i=0; i<$TRAINING_EPOCHES; i++)); do
    NUM_EPOCHES=$(($i+$INIT_EPOCHES))
    NUM_STEPS=$(($NUM_EPOCHES*$STEPS_PER_EPOCH))
    CURR_DIR="epoch-$NUM_EPOCHES"

    CURR_WEIGHTS="$WORKSPACE/model/s$NUM_STEPS.txt"

    safe_mkdir $CURR_DIR

    # step1: self-play
    ENGINE_PLAY_CMD="./$ENGINE_NAME --mode selfplay --noise --random-moves 10 --playouts 200 --komi 9 --board-size 7 --komi-variant 2"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --parallel-games 16 --batch-size 8"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --cache-memory-mib 400 --early-symm-cache"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --target-directory $CURR_DIR --weights $CURR_WEIGHTS --num-games $GAMES_PER_EPOCH"
    $ENGINE_PLAY_CMD


    # step2: shuffle data
    INPUT_DIR="$CURR_DIR/data"
    OUPUT_DIR="$CURR_DIR/data-gz"
    NUM_CORES=1

    safe_mkdir $OUPUT_DIR

    SHUFFLE_CMD="python3 train/shuffle.py -n $NUM_CORES --input-dir $INPUT_DIR --output-dir $OUPUT_DIR"
    $SHUFFLE_CMD

    # step3: move training data
    for ((j=0; j<$DELAY_BUF_EPOCHES; j++)); do
        T_EPOCH=$(($NUM_EPOCHES-$j))
        if [ $T_EPOCH -ge 0 ]; then
            DATA_EPOCH="$TRAINING_SET_DIR/data-$T_EPOCH"
            GZ_EPOCH="epoch-$T_EPOCH"

            safe_mkdir $DATA_EPOCH
            mv "$GZ_EPOCH/data-gz" $DATA_EPOCH
        fi
    done


    # step4: start training
    TRAIN_CMD="python3 train/parser.py -j train/setting.json"
    $TRAIN_CMD


    # step5: transfer the current model
    NEXT_NUM_STEPS=$(($NUM_STEPS+$STEPS_PER_EPOCH))
    TRANSFER_CMD="python3 train/transfer.py -j train/setting.json -n $WORKSPACE/model/s$NEXT_NUM_STEPS"
    $TRANSFER_CMD


    # step6: remove training data
    for ((j=0; j<$DELAY_BUF_EPOCHES; j++)); do
        T_EPOCH=$(($NUM_EPOCHES-$j))
        if [ $T_EPOCH -ge 0 ]; then
            DATA_EPOCH="$TRAINING_SET_DIR/data-$T_EPOCH"
            GZ_EPOCH="epoch-$T_EPOCH/data-gz"

            mv "$DATA_EPOCH/data-gz" "$GZ_EPOCH"
            safe_rmdir $DATA_EPOCH
        fi
    done
done
