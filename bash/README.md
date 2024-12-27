# Reinforcement Learning

## Requirements

* Be sure that you had built the engine. The engine should be in the ```build``` directory. Recommend to use the ```-DUSE_ZLIB=1``` option to save your disk memory.
* PyTorch 1.x (for python)
* NumPy (for python)

## Performance

Use the default setting in the bash directory. The network will at least reach strong amateur level in 1 ~ 2 weeks on 19x19 with a RTX 2070s computer.

## Simple Usage

There are two bash files. The ```setup.sh``` will copy the engine and training script to the current directory. Then executing ```simple.sh``` and start simple synchronized loop. And default will use all GPUs.

    $ cp -r bash selfplay-course
    $ cd selfplay-course
    $ bash setup.sh -s ..
    $ bash simple.sh

The ```simple.sh``` will do the infinite loop. If you want to halt the loop, you need to create a ```kill.txt``` file and wait for end of this round.

    $ touch kill.txt

Maybe you have a powerful computer with multi-GPUs but don't want to use all GPUs. You may add the option ```-g``` to execute the self-play and training on the specific GPU.

    $ bash simple.sh -g 1    # run it on the 2nd GPU

Or select specific multi-GPUs.

    $ bash simple.sh -g 1 -g 2    # run it on the 2nd and 3rd GPU

## Sample Configuration File

The sample directory includes some enigne self-play Configuration files. The ```sample/full-gumbel-p16.txt``` will do full Gumbel learning with 16 playouts. The ```sample/full-alphazero-p400.txt``` will do full AlphaZero learning with 400 playouts. You may simply reuse the files for customization learning. And please see this [section](./CONFIG.md). It explains the configuration and training parameters.
