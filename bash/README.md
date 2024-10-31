# Reinforcement Learning

## Requirements

* Be sure that you had built the engine. The engine should be in the ```build``` directory. Recommend to use the ```-DUSE_ZLIB=1``` option.
* PyTorch 1.x (for python)
* NumPy (for python)

## Note

Use the default setting in the bash directory. The network will reach strong amateur level in 1 ~ 2 weeks on 19x19 with the RTX 2070s computer.

## Simple Usage

There are two bash files. The ```setup.sh``` will do the initialization. Copy the training script and engine to this directory. The ```simple.sh``` will do the self-play and trainig loop.

    $ cp -r bash selfplay-course
    $ cd selfplay-course
    $ bash setup.sh -s ..
    $ bash simple.sh

The ```simple.sh``` will do the infinite loop. If you want to stop the loop, you need to create a kill file and wait for end of this round.

    $ touch kill.txt

## Sample Configuration File

The sample directory includes some enigne self-play Configuration files. The ```sample/full-gumbel-p16.txt``` will do full Gumbel learning with 16 playouts. The ```sample/full-alphazero-p400.txt``` will do full AlphaZero learning with 400 playouts. You may simply reuse the files for customization learning. And please see this [section](./CONFIG.md). It explains the configuration and training parameters.
