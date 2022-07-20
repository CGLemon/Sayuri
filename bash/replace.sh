usage()
{
    echo "usage: -e <int> | last epoches"
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
        rm -rf $1
        echo "rm -rf $1"
    fi
}

if [ "$1" = "-h" ] || [ $# = 0 ]; then
    usage
    exit 1
fi

while getopts e: flag
do
    case "${flag}" in
        e) last_epoches=${OPTARG};;
    esac
done

# basic parameters
WORKSPACE="workspace"
TRAINING_SET_DIR="train-dir"
LAST_STEPS_FILE="$WORKSPACE/last_steps.txt"

DELAY_BUF_EPOCHES=20
LAST_EPOCHES=$last_epoches

safe_rmdir $WORKSPACE
safe_mkdir $WORKSPACE

# step1: move training data
for ((j=0; j<$DELAY_BUF_EPOCHES; j++)); do
    T_EPOCH=$(($LAST_EPOCHES-$j))
    if [ $T_EPOCH -ge 0 ]; then
        DATA_EPOCH="$TRAINING_SET_DIR/data-$T_EPOCH"
        GZ_EPOCH="epoch-$T_EPOCH"

        safe_mkdir $DATA_EPOCH
        mv "$GZ_EPOCH/data-gz" $DATA_EPOCH
    fi
done

# step2: start training
TRAIN_CMD="python3 torch/parser.py -j torch/setting.json"
$TRAIN_CMD


# step3: transfer the current model
if [ -f $LAST_STEPS_FILE ]; then
    NUM_STEPS=$( cat $LAST_STEPS_FILE )
fi
TRANSFER_CMD="python3 torch/transfer.py -j torch/setting.json -n $WORKSPACE/model/s$NUM_STEPS"
$TRANSFER_CMD


# step4: remove training data
for ((j=0; j<$DELAY_BUF_EPOCHES; j++)); do
    T_EPOCH=$(($LAST_EPOCHES-$j))
    if [ $T_EPOCH -ge 0 ]; then
        DATA_EPOCH="$TRAINING_SET_DIR/data-$T_EPOCH"
        GZ_EPOCH="epoch-$T_EPOCH/data-gz"

        mv "$DATA_EPOCH/data-gz" "$GZ_EPOCH"
        safe_rmdir $DATA_EPOCH
    fi
done
