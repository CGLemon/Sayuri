# Sayuri

## Let's ROCK

Sayuri is a super human level 9x9 go bot.

![step_one](https://github.com/CGLemon/Sayuri/blob/master/img/sayuri-art.jpg)

[https://medibang.com/picture/r32007181509071270014632767/?locale=zh_TW](https://medibang.com/picture/r32007181509071270014632767/?locale=zh_TW)

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

## Optinal Building

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

1. Support for sabaki analyzing mode.
2. Support for handicap game.
3. Support for variant komi.
4. Support for different board size(but much weaker).
5. Lock-free SMP MCTS.
6. Acceleration by mutil-CPU and Nvidia GPU.
7. Predict current side winrate.
8. Predict current score lead.
9. Predict the death strings.

## LICENSE

GNU GPL version 3 section 7
