# Sayuri

![step_one](https://github.com/CGLemon/Sayuri/blob/master/img/sayuri-art.jpg)

## Let's ROCK!

Sayuri is a super human level 9x9 go bot and supports for full GTP protocol.

## Prepare

Following library is required.

1. CMake(>=3.9)

Following library are optional.

1. OpenBLAS
2. CUDA
3. cuDNN

## Build(Ubuntu/MacOS)

The program is only available on Ubuntu/MacOS.

    $ git clone https://github.com/CGLemon/Sayuri
    $ cd Sayuri
    $ mkdir build && cd build
    $ cmake ..
    $ make -j

## Optional Building

Accelerate the network on CPU. OpenBlas is required. OpenBlas is significantly faster than built-in blas.

    $ cmake .. -DBLAS_BACKEND=OPENBLAS

Accelerate the network by GPU. CUDA is required. It will be faster than cuDNN in only one batch size.

    $ cmake .. -DBLAS_BACKEND=CUDA

Accelerate the network by GPU. CUDA and cuDNN are required. It will be faster than CUDA-only in large batch size.

    $ cmake .. -DBLAS_BACKEND=CUDNN

## Weights

You may download the weights from my [google drive](https://drive.google.com/file/d/1tZJ_9ZY_OMDZHDxaELTtPa1bmvIQFGtk/view?usp=sharing). The weights size is 15 blocks and 192 filters(around 155MB).


## Engine Options

Here are some useful options which you may set.

    --weights, -w: File with network weights.
    
    $ ./Sayuri -w <weights file>
    
    
    --playouts, -p: Set the playouts limit. Bigger is stronger.
    
    $ ./Sayuri -p 1600
    
    
    --threads, -t: Set the search threads. Bigger is faster.
    
    $ ./Sayuri -t 4
    
    
    --batchzie, -b: Set the network batch size. Bigger is faster.
    
    $ ./Sayuri -b 2
    
    
    --analysis-verbose, -a: Output more search verbose.
    
    $ ./Sayuri --analysis-verbose
    
    
    --quiet, -q: Disable all diagnostic verbose.

    $ ./Sayuri --quiet
    
    
    --help, -h: Display the more options.

    $ ./Sayuri --help
    
    
    Exmaple:
    
    $ ./Sayuri -w <weights file> -t 4 -b 2 -p 1600

## Features

* Support for sabaki analyzing mode.
* Support for handicap game.
* Support for variant komi.
* Support for different board size(but much weaker).
* Lock-free SMP MCTS.
* Acceleration by multi-CPU and Nvidia GPU.
* Predict current side winrate.
* Predict current score lead.
* Predict the death strings.

## LICENSE

GNU GPL version 3 section 7

## Other
* The picture is from [here](https://medibang.com/picture/r32007181509071270014632767/?locale=zh_TW).
