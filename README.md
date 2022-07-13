<div id="sayuri-art" align="center">
    <br/>
    <img src="https://github.com/CGLemon/Sayuri/blob/master/img/sayuri-art.PNG" alt="Sayuri Art" width="768"/>
    <h3>Sayuri</h3>
</div>

## Let's ROCK!

Sayuri is a GTP-compliant go engine which supports variable komi and board size. Strongly inspired by Leela Zero and Kata Go. Based on Deep Convolutional Neural Network, Monte Carlo Tree Search and other techniques.

## Requirements

* Ubuntu or MacOS only
* GCC, Clang which C++14 compiler
* CMake 3.15 or later
* Optional: Eigen or OpenBLAS library
* Optional: CUDA 10.x or 11.x library (GCC 7.x passed)
* Optional: cuDNN 7.x or 8.x library
* Optional: Fast Float library

## Default Compiling (Ubuntu or MacOS)

    $ git clone https://github.com/CGLemon/Sayuri
    $ cd Sayuri
    $ git submodule update --init --recursive
    $ mkdir build && cd build
    $ cmake ..
    $ make -j

## Optional Compiling

Accelerate the network by CPU. OpenBlas or Eigen are required. OpenBlas and Eigen are significantly faster than built-in blas. OpenBlas is recommended on MacOS.

    $ cmake .. -DBLAS_BACKEND=OPENBLAS

or

    $ cmake .. -DBLAS_BACKEND=EIGEN

Accelerate the network by GPUs. CUDA is required. It will be faster than cuDNN in only one batch size case.

    $ cmake .. -DBLAS_BACKEND=CUDA

Accelerate the network by GPUs. CUDA and cuDNN are required. It will be faster than CUDA-only in multi batch size case.

    $ cmake .. -DBLAS_BACKEND=CUDNN


Accelerate to load the network file. Fast Float library is required.

    $ cmake .. -USE_FAST_PARSER=1


## Weights and Book

You may download the weights file and opening book from my [google drive](https://drive.google.com/drive/folders/1SgPL3Eyhllr6BCDyi_7D8LnOUYxPAAxQ?usp=sharing). The weights size is 15 blocks and 192 filters. The opening book is human-like book, trained on profession games. Force the Sayuri to play variable opening moves. It is just fun for playing.

## Engine Arguments

Here are some useful arguments which you may need.

| Arguments               | Param  | Description                                    |
| :---------------------- | :----- | :--------------------------------------------- |
|  --weights, -w          | string | File with network weights.                     |
|  --book, -b             | string | File with opening book.                        |
|  --playouts, -p         | int    | The number of maximum playouts.                |
|  --const-time           | int    | Const time of search in seconds.               |
|  --threads, -t          | int    | The number of threads used.                    |
|  --batch-size, -b       | int    | The number of batches for a single evaluation. |
|  --gpu, -g              | int    | Select a specific GPU device.                  |
|  --resign-threshold, -r | float  | Resign when winrate is less than x.            |
|  --analysis-verbose, -a | None   | Output more search diagnostic verbose.         |
|  --quiet, -q            | None   | Disable all diagnostic verbose.                |
|  --ponder               | None   | Thinking on opponent's time.                   |
|  --friendly-pass        | None   | Do pass move if the engine wins the game.      |
|  --reuse-tree           | None   | Will reuse the sub-tree.                       |
|  --help, -h             | None   | Show the more arguments.                       |
    

Default setting: will select reasonable thread and batch size, 15 seconds per move, all GPU devices

    $ ./Sayuri -w <weights file>

Example setting 1: 4 thread, 2 batches and 12800 playouts
    
    $ ./Sayuri -w <weights file> -t 4 -b 2 -p 12800

Example setting 2: quickly and friendly game
    
    $ ./Sayuri -w <weights file> -t 1 -b 1 --const-time 1 --friendly-pass --reuse-tree

Example setting 3: direct policy output 

    $ ./Sayuri -w <weights file> -t 1 -b 1 -p 1

Example setting 4: use the GPU 0 and GPU 2

    $ ./Sayuri -w <weights file> --gpu 0 --gpu 2


## Generate Opening Book

You need to collect enough SGF games (at least over 10000 games). Then, go to the GTP mode and enter follow command. Wait some time to generate a new opening book.

    genbook <SGF file> <output name>

## User Interface

Sayuri is not complete engine. You need a graphical interface for playing with her. She supports any GTP (version 2) interface application. [Sabaki](https://sabaki.yichuanshen.de/) and [GoGui](https://github.com/Remi-Coulom/gogui) are recommended. 

* Sabaki analysis mode

![sabaki-sample01](https://github.com/CGLemon/Sayuri/blob/master/img/sabaki-sample01.png)

* GoGui analysis command

![gogui-sample01](https://github.com/CGLemon/Sayuri/blob/master/img/gogui-sample01.png)

## Features

* Support sabaki analysis mode.
* Support some GoGui analysis commands.
* Support handicap game.
* Support variable komi.
* Support variable board size (from 7x7 to 19x19).
* Lock-free SMP MCTS.
* Acceleration by multi-core processor and multi-Nvidia GPU.
* Predict the current side winrate and draw-rate.
* Predict the current side score lead.
* Predict the death strings.
* Reuse the sub-tree.

## Todo

* Support half-float.
* Support NHWC format.
* Support distributed computation.
* Store the networks as binary file.
* Including pattern system (should finish it in beta version).

## Other Linkings

* Go Text Protocol, [https://www.gnu.org/software/gnugo/gnugo_19.html](https://www.gnu.org/software/gnugo/gnugo_19.html)
* Leela Zero, [https://github.com/leela-zero/leela-zero](https://github.com/leela-zero/leela-zero)
* 開發日誌, [https://hackmd.io/zulj1rvhQROsB7U3poEjQg?view](https://hackmd.io/zulj1rvhQROsB7U3poEjQg?view)

## License

The code is released under the GPLv3, except for threadpool.h, filesystem.h, filesystem.cc, cppattributes.h, mm.h, mm.cc, Eigen and Fast Float, which have specific licenses mentioned in those files.

## Contact

cglemon000@gmail.com (Hung-Zhe Lin)
