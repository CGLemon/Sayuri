#!/bin/bash

usage()
{
    echo "usage: -h, --help      | dump this verbose"
    echo "usage: -g, --gpu <int> | select the specific GPU(s) device"
    exit 1
}

init_params()
{
    WORKSPACE="workspace"
    WEIGHTS_DIR="weights"
    SELFPLAY_DIR="selfplay"
    KILL_FILE="kill.txt"
    CONFIG_FILE="selfplay-config.txt"
    ENGINE_NAME="sayuri"
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

create_workspace()
{
    mkdir -p $SELFPLAY_DIR
    mkdir -p $WEIGHTS_DIR
}

main_loop()
{
    if ! [ -f $CONFIG_FILE ]; then
        echo "The engine config file does not exist. Exit!"
        exit 1
    fi

    create_workspace

    while true
    do
        # Stop the loop if we find the kill file.
        if [ -f $KILL_FILE ]; then
            echo "Find the kill file. Stop the self-play loop."
            break
        fi

        # Do the self-play.
        ENGINE_PLAY_CMD="./$ENGINE_NAME --mode selfplay --config $CONFIG_FILE"
        ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD $GPU_FLAG"
        ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --target-directory $SELFPLAY_DIR"
        ENGINE_PLAY_CMD="$ENGINE_PLAY_CMD --weights-dir $WEIGHTS_DIR"

        echo $ENGINE_PLAY_CMD
        $ENGINE_PLAY_CMD
    done
}

while :; do
    case $1 in
        -h|--help) shift; usage
        ;;
        -c|--config) shift; CONFIG_FILE=$1
        ;;
        -g|--gpu) shift; GPU_LIST="$GPU_LIST $1";
        ;;
        -k|--kill) shift; KILL_FILE=$1;
        ;;
        "") break
        ;;
        *) echo "Unknown argument: $1"; usage
        ;;
    esac
    shift
done


gather_gpu
main_loop

