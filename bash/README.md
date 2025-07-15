# Reinforcement Learning

## Requirements

* Ensure that the engine has been built and is located in the build directory. It is recommended to use the -DUSE_ZLIB=1 option during compilation to reduce disk space usage.
* PyTorch 1.x or PyTorch 2.x (Python)
* NumPy (Python)

## Performance

With the default settings in the ```bash``` directory, the network can generally reach at least strong amateur level on 19×19 boards within 1 to 2 weeks when training on a single RTX 2070 Super GPU.

## Simple Usage

There are two main bash scripts available. The ```setup.sh``` script copies the engine and training scripts into the current directory. Afterwards, you can run simple.sh to start a basic synchronized training loop, which by default utilizes all available GPUs.

    $ cp -r bash selfplay-course
    $ cd selfplay-course
    $ bash setup.sh -s ..
    $ bash simple.sh

The ```simple.sh``` script will run in an infinite loop. To safely stop the loop, create a ```kill.txt``` file and the process will terminate after completing the current round.

    $ touch kill.txt

If you have a multi-GPU system and wish to limit the GPUs used, you can specify which GPUs to use with the -g option:

    $ bash simple.sh -g 1    # run it on the 2nd GPU

To use multiple specific GPUs:

    $ bash simple.sh -g 1 -g 2    # run it on the 2nd and 3rd GPU

## Sample Configuration File

The sample directory includes several example self-play engine configuration files:

* ```sample/full-gumbel-p16.txt``` runs full Gumbel training with 16 playouts.
* ```sample/full-alphazero-p400.txt``` runs full AlphaZero-style training with 400 playouts.

These sample files can be reused and modified for customized training setups. For a detailed explanation of all configuration and training parameters, please refer to the [CONFIG.md](./CONFIG.md) section.


