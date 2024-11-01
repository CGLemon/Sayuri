#!/bin/bash

function usage()
{
    echo "usage: -h, --help        | dump this verbose"
    echo "usage: -n, --no-loop     | execute the code only once"
    echo "usage: -k, --kill <path> | kill file path"
    exit 1
}

function main_loop()
{
    # Copy the self-play weights from here.
    WEIGHTS_SOURCE="$WORKSPACE/swa"

    mkdir -p $WORKSPACE
    mkdir -p $WEIGHTS_SOURCE
    mkdir -p $WEIGHTS_DIR

    while true
    do
        # Stop the loop if we find the kill file.
        if [ -f $KILL_FILE ]; then
            echo "Find the kill file. Stop the self-play loop."
            break
        fi

        # Copy the weights to the target directory.
        srcweights=$(ls -At $WEIGHTS_SOURCE | head -n 1)
        tgtweights=$(ls -At $WEIGHTS_DIR | head -n 1)

        # TODO: We should implement gate engine to pick self-play weights.
        if [ "$srcweights" == "$tgtweights" ]; then
            sleep 1
        else
            cp -p $WEIGHTS_SOURCE/$srcweights $WEIGHTS_DIR
        fi

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
        -k|--kill) shift; KILL_FILE=$1;
        ;;
        "") break
        ;;
        *) echo "Unknown argument: $1"; usage
        ;;
    esac
    shift
done

main_loop
