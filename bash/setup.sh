#!/bin/bash

usage()
{
    echo "usage: -s <path> | source directory"
    exit 1
}

SOURCE_DIR=""

while :; do
    case $1 in
        -h|--help) shift; usage
        ;;
        -s) shift; SOURCE_DIR="$1"
        ;;
        "") break
        ;;
        *) echo "Unknown argument: $1"; usage
        ;;
    esac
    shift
done

if [ "$SOURCE_DIR" = "" ]; then
    usage
fi

ENGINE_PATH="$SOURCE_DIR/build/sayuri"
TORCH_PATH="$SOURCE_DIR/train/torch"

if  [ ! -x ${ENGINE_PATH} ]; then
    echo "The ${ENGINE_PATH} does not exists!"
    exit 1
fi

cp "$ENGINE_PATH" "."
cp -r "$TORCH_PATH" "."
echo "copy done!"
