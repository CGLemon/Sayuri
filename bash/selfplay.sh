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

usage()
{
    echo "usage: -g, --gpu <int> | select the specific GPU(s) device"
    exit 1
}

gather_gpu()
{
    if [ "$GPU_LIST" = "" ]; then
        GPU_LIST=$(echo $(nvidia-smi -L | wc -l) | awk '{for(i=0;i<$1;i++)printf "%d ",i}')
    fi

    NUM_GPU=0
    for i in $GPU_LIST; do NUM_GPU=$((NUM_GPU + 1)); CUDA_DEVICES+="$i,"; GPU_FLAG+="-g $i " ; done
    CUDA_DEVICES=${CUDA_DEVICES::-1}
    GPU_FLAG=${GPU_FLAG::-1}
}

while :; do
    case $1 in
        -h|--help) shift; usage
        ;;
        -g|--gpu) shift; GPU_LIST="$GPU_LIST $1";
        ;;
        "") break
        ;;
        *) echo "Unknown argument: $1"; usage
        ;;
    esac
    shift
done

gather_gpu

mkdir -p $SELFPLAY_DIR
mkdir -p $WORKSPACE
mkdir -p $WEIGHTS_DIR

while true
do
    # Do the self-play.
    ENGINE_PLAY_CMD="./$ENGINE_NAME --mode selfplay --config $CONFIG_FILE"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD $GPU_FLAG"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --target-directory $SELFPLAY_DIR"
    ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --weights-dir $WEIGHTS_DIR"

    echo $ENGINE_PLAY_CMD
    $ENGINE_PLAY_CMD

    # Train a new model.
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 torch/main.py -j ${SETTING_FILE}

    # Stop the loop if we find the kill file.
    if [ -f $KILL_FILE ]; then 
        echo "Find the kill file. Stop the self-play loop."
        rm $KILL_FILE
        break
    fi
done
