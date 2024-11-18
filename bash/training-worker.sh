#!/bin/bash

function usage()
{
    echo "usage: -h, --help             | dump this verbose"
    echo "usage: -n, --no-loop          | execute the code only once"
    echo "usage: -s, --setting <path>   | training setting file path"
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

    gather_gpu
    mkdir -p $WORKSPACE

    while true
    do
        # Stop the loop if we find the kill file.
        if [ -f $KILL_FILE ]; then
            echo "Find the kill file. Stop the self-play loop."
            break
        fi

        # Train a new model.
        CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 torch/train.py -j ${SETTING_FILE} -w ${WORKSPACE}

        if (($EXECUTE_LOOP == 0)); then
            break
        fi
    done
}

source ./default_param.sh

while :; do
    case $1 in
        -h|--help) shift; usage
        ;;
        -n|--no-loop) shift; EXECUTE_LOOP=0
        ;;
        -s|--setting) shift; SETTING_FILE=$1
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

