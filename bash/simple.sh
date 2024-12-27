#!/bin/bash

function usage()
{
    echo "usage: -h, --help             | dump this verbose"
    echo "usage: -s, --setting <path>   | training setting file path"
    echo "usage: -c, --config <path>    | self-paly setting file path"
    echo "usage: -g, --gpu <int>        | select the specific GPU(s) device"
    echo "usage: -k, --kill <path>      | kill file path"
    echo "usage: -w, --workspace <path> | workspace path"
    exit 1
}

function gather_gpu()
{
    if [ "$GPU_LIST" = "" ]; then
        GPU_LIST=$(echo $(nvidia-smi -L | wc -l) | awk '{for(i=0;i<$1;i++)printf "%d ",i}')
    fi

    NUM_GPUS=0
    for i in $GPU_LIST; do NUM_GPUS=$((NUM_GPUS + 1)); CUDA_DEVICES+="$i,"; GPU_FLAG+="-g $i " ; done
    CUDA_DEVICES=${CUDA_DEVICES::-1}
    GPU_FLAG=${GPU_FLAG::-1}
}

function main_loop()
{
    if ! [ -f $SETTING_FILE ]; then
        echo "The training setting file does not exist. Exit!"
        exit 1
    fi
    if ! [ -f $CONFIG_FILE ]; then
        echo "The engine config file does not exist. Exit!"
        exit 1
    fi

    gather_gpu

    while true
    do
        # Stop the loop if we find the kill file.
        if [ -f $KILL_FILE ]; then
            echo "Find the kill file. Stop the self-play loop."
            rm $KILL_FILE
            break
        fi

        NULL_KILL_FILE="null"
        bash selfplay-worker.sh -c $CONFIG_FILE -k $NULL_KILL_FILE $GPU_FLAG --no-loop
        bash training-worker.sh -s $SETTING_FILE -k $NULL_KILL_FILE -w $WORKSPACE $GPU_FLAG --no-loop
        bash gate-worker.sh -k $NULL_KILL_FILE -w $WORKSPACE --no-loop
    done
}

source ./default_param.sh

while :; do
    case $1 in
        -h|--help) shift; usage
        ;;
        -s|--setting) shift; SETTING_FILE=$1
        ;;
        -c|--config) shift; CONFIG_FILE=$1
        ;;
        -g|--gpu) shift; GPU_LIST="$GPU_LIST $1";
        ;;
        -k|--kill) shift; KILL_FILE=$1;
        ;;
        -w|--workspace) shift; WORKSPACE=$1;
        ;;
        "") break
        ;;
        *) echo "Unknown argument: $1"; usage
        ;;
    esac
    shift
done

main_loop
