#!/bin/bash

# basic parameters
WORKSPACE="workspace"
LAST_STEPS_FILE="$WORKSPACE/last_steps.txt"
SETTING_FILE="torch/selfplay-setting.json"

safe_mkdir()
{
    if [ ! -d "$1" ]; then
        mkdir $1
    fi
}

# step1: remove the old data
NOW=$(date +"%Y-%m-%d:%H-%M-%S")
if [ -d $WORKSPACE ]; then
    mv $WORKSPACE $NOW
fi
safe_mkdir $WORKSPACE

# step2: start training
TRAIN_CMD="python3 torch/parser.py -j $SETTING_FILE"
$TRAIN_CMD

# step3: transfer the current model
if [ -f $LAST_STEPS_FILE ]; then
    NUM_STEPS=$( cat $LAST_STEPS_FILE )
fi
TRANSFER_CMD="python3 torch/transfer.py -j torch/setting.json -b -n $WORKSPACE/model/s$NUM_STEPS"
$TRANSFER_CMD

