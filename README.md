# Sayuri

![picture](https://github.com/CGLemon/Sayuri/blob/master/img/sayuri-art.jpg)

## Let's ROCK!

Sayuri is a super human level 9x9 GTP go engine. Strongly inspired by Leela Zero and Kata Go. Based on Deep Neural Network, Monte Carlo Tree Search and other powerful skills.

## Requirements

Following library is required.

1. CMake(>=3.15)

Following library are optional.

1. Eigen
2. OpenBLAS
3. CUDA(<=10.x)
4. cuDNN(<=8.x)

## Build(Ubuntu/MacOS)

The program is only available on Ubuntu/MacOS.

    $ git clone https://github.com/CGLemon/Sayuri
    $ git submodule update --init --recursive
    $ cd Sayuri
    $ mkdir build && cd build
    $ cmake ..
    $ make -j

## Optional Building

Accelerate the network by CPU. OpenBlas and Eigen are required. OpenBlas and Eigen are significantly faster than built-in blas. OpenBlas is recommended on MacOS.

    $ cmake .. -DBLAS_BACKEND=OPENBLAS

or

    $ cmake .. -DBLAS_BACKEND=EIGEN

Accelerate the network by GPUs. CUDA is required. It will be faster than cuDNN in only one batch size.

    $ cmake .. -DBLAS_BACKEND=CUDA

Accelerate the network by GPUs. CUDA and cuDNN are required. It will be faster than CUDA-only in multi batch size.

    $ cmake .. -DBLAS_BACKEND=CUDNN

## Weights

You may download the weights from my [google drive](https://drive.google.com/file/d/1vkx886BV55ujwJatuzVm19Jok18M-3Jj/view?usp=sharing). The weights size is 15 blocks and 192 filters(around 170MB).

## Engine Arguments

Here are some useful arguments which you may need.

    --weights, -w: File with network weights.
    
    $ ./Sayuri -w <weights file>
    
    
    --playouts, -p: Set the playouts limit. Bigger is stronger.
    
    $ ./Sayuri -p 1600
    
    
    --threads, -t: Set the search threads. Bigger is faster. The default setting will select a reasonable number.
    
    $ ./Sayuri -t 4
    
    
    --batch-size, -b: Set the network batch size. Bigger is faster. The default setting will select a reasonable number.
    
    $ ./Sayuri -b 2
    
    
    --resign-threshold, -r: Resign when winrate is less than x.
    
    $ ./Sayuri -r 0.2
    
    
    --analysis-verbose, -a: Output more search diagnostic verbose.
    
    $ ./Sayuri --analysis-verbose
    
    
    --quiet, -q: Disable all diagnostic verbose (some GTP GUI need to disable them).
    
    $ ./Sayuri --quiet
    
    
    --ponder: Thinking on opponent's time.
    
    $ ./Sayuri --ponder
    
    
    --help, -h: Display the more options.
    
    $ ./Sayuri --help
    
    
    
    Exmaple:
    
    $ ./Sayuri -w <weights file> -t 4 -b 2 -p 1600

## User Interface

Sayuri supports for any GTP interface application. [Sabaki](https://sabaki.yichuanshen.de/) is recommended.

## Features

* Support for sabaki analyzing mode.
* Support for handicap game.
* Support for variant komi.
* Support for variant board size.
* Lock-free SMP MCTS.
* Acceleration by multi-core processor and multi-Nvidia GPU.
* Predict current side winrate.
* Predict current side score lead.
* Predict the death strings.

## Todo

* Reuse the tree.
* optimize the training pipe.
* Support for half-float.
* Support for NHWC format.
* Support for distributed computation.
* Including pattern system.
* Training bigger size network.

## LICENSE

GNU GPL version 3 section 7

## Other
* The picture is from [here](https://i.ytimg.com/vi/EzgE7CbTs3o/maxresdefault.jpg).
