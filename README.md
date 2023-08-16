
<div id="sayuri-art" align="center">
    <br/>
    <img src="./img/sayuri-art.PNG" alt="Sayuri Art" width="768"/>
    <h3>Sayuri</h3>
</div>

## Let's ROCK!

Sayuri is a GTP-compliant go engine based on Deep Convolutional Neural Network and Monte Carlo Tree Search. She is strongly inspired by Leela Zero and KataGo. The board data structure, search algorithm and network format are borrowed from Leela Zero in the beginning. Current version follows the KataGo research, the engine supports variable komi and board size now. Some methods you may see my HackMD articles (in chinese).

* [開發日誌](https://hackmd.io/@yrHb-fKBRoyrKDEKdPSDWg/BJgfay0Yc)
* [AlphaZero 之加速演算法實作](https://hackmd.io/@yrHb-fKBRoyrKDEKdPSDWg/HJI9_p70i)

## Requirements

* Ubuntu, MacOS or Windows
* GCC, Clang, must support C++14 or higher
* CMake 3.15 or higher
* Optional: Eigen or OpenBLAS library
* Optional: CUDA 10.x - 12.x library
* Optional: cuDNN 7.x or 8.x library
* Optional: zlib library

## Default Compiling (Ubuntu or MacOS)

    $ git clone https://github.com/CGLemon/Sayuri
    $ cd Sayuri
    $ git submodule update --init --recursive
    $ mkdir build && cd build
    $ cmake ..
    $ make -j

## Cmake Option

Accelerate the network forwarding pipe by CPU. OpenBLAS or Eigen are required. OpenBLAS and Eigen are significantly faster than built-in blas. OpenBLAS is recommended on MacOS.

    $ cmake .. -DBLAS_BACKEND=OPENBLAS

or

    $ cmake .. -DBLAS_BACKEND=EIGEN

Accelerate the network forwarding pipe by GPUs. CUDA is required. It should be as faster as cuDNN backend.

    $ cmake .. -DBLAS_BACKEND=CUDA

Accelerate the network forwarding pipe by GPUs. CUDA and cuDNN are both required. This backend is much steady than CUDA-only backend.

    $ cmake .. -DBLAS_BACKEND=CUDNN

Compile a bigger board size version. Set it as 0 to disable this option. 

    $ cmake .. -DBOARD_SIZE=25

Disable the FP16 CUDA code if your CUDA version doesn't support for it.

    $ cmake .. -DDISABLE_FP16=1

Compress the training data file. It can save many memory usage in the self-play process.

    $ cmake .. -DUSE_ZLIB=1

## Windows Version (Experiment)

1. Download the Visual Studio.
2. Download the MinGW from [here](https://github.com/mstorsjo/llvm-mingw).
3. Clone the github repo and compile it.

        $ git clone https://github.com/CGLemon/Sayuri
        $ cd Sayuri
        $ git submodule update --init --recursive
        $ cd src
        $ g++ -std=c++14 -ffast-math -I . -lpthread *.cc utils/*.cc accuracy/*.cc game/*.cc mcts/*.cc neural/*.cc neural/blas/*.cc neural/cuda/*.cc pattern/*.cc selfplay/*.cc -o Sayuri -O3 -DNDEBUG -DWIN32 -I ../third_party/Eigen -DUSE_BLAS -DUSE_EIGEN


## Weights and Others.

Download the last v0.6 weights [here](https://drive.google.com/drive/folders/1nawHAKHTBKEpLcizaVrK4GVDSIuVqJ-Q?usp=sharing) and see the current RL progression [here](https://hackmd.io/@yrHb-fKBRoyrKDEKdPSDWg/HJew5OFci). If you want to use the old network, please use the v0.5 engine.

## Engine Arguments

Here are some useful arguments which you may need.

| Arguments               | Type   | Description                                    |
| :---------------------- | :----- | :--------------------------------------------- |
|  --weights, -w          | string | File with network weights.                     |
|  --patterns             | string | File with patterns.                            |
|  --book                 | string | File with opening book.                        |
|  --playouts, -p         | int    | The number of maximum playouts.                |
|  --const-time           | int    | Const time of search in seconds.               |
|  --threads, -t          | int    | The number of threads used.                    |
|  --batch-size, -b       | int    | The number of batches for a single evaluation. |
|  --gpu, -g              | int    | Select a specific GPU device.                  |
|  --resign-threshold, -r | float  | Resign when winrate is less than x.            |
|  --analysis-verbose, -a | None   | Output more search diagnostic verbose.         |
|  --quiet, -q            | None   | Disable all diagnostic verbose.                |
|  --ponder               | None   | Thinking on opponent's time.                   |
|  --friendly-pass        | None   | Play pass move if the engine won the game.     |
|  --reuse-tree           | None   | Will reuse the sub-tree.                       |
|  --help, -h             | None   | Show the more arguments.                       |
    
<br/>

Default setting: will select reasonable threads and batch size, 10 seconds per move, all GPU devices

    $ ./Sayuri -w <weights file>

Example setting 1: 4 threads, 2 batches and 12800 playouts
    
    $ ./Sayuri -w <weights file> -t 4 -b 2 -p 12800

Example setting 2: quickly and friendly pass game
    
    $ ./Sayuri -w <weights file> -t 1 -b 1 --const-time 1 --friendly-pass

Example setting 3: set 0 playouts, directly policy output 

    $ ./Sayuri -w <weights file> -t 1 -b 1 -p 0

Example setting 4: use the GPU 0 and GPU 2

    $ ./Sayuri -w <weights file> --gpu 0 --gpu 2

Example setting 5: disable the network forwarding pipe, arond 5k on 9x9, 10k on 19x19. The ```--lcb-reduction``` should be set as ```1```

    $ ./Sayuri --patterns <patterns file> --lcb-reduction 1 --no-dcnn

## Graphical Interface

Sayuri is not complete engine. You need a graphical interface for playing with her. She supports any GTP (version 2) interface application. [Sabaki](https://sabaki.yichuanshen.de/) and [GoGui](https://github.com/Remi-Coulom/gogui) are recommended that because Sayuri supports some specific analysis commands. 

* Sabaki analysis mode

![sabaki-sample01](./img/sabaki-sample01.png)

* GoGui analysis commands

![gogui-sample01](./img/gogui-sample01.png)

## Analysis Commands

The analysis Commands are useful on the modern GTP interface tool, like Sabaki. It shows the current winrate, best move and the other informations. The engine supports the following GTP analysis commands.

  * `analyze, genmove_analyze [player (optional)] [interval (optional)] ...`
      * The behavior is same as ```lz-analyze```, ```lz-genmove_analyze```.

  * `lz-analyze, lz-genmove_analyze [player (optional)] [interval (optional)] ...`
      * Extension GTP commands of ```lz-analyze``` and ```lz-genmove_analyze```. Support the ```info```, ```move```, ```visits```, ```winrate```, ```prior```, ```lcb```, ```order```, ```pv```, ```scoreLead``` labels. More detail to see [KataGo GTP Extensions](https://github.com/lightvector/KataGo/blob/master/docs/GTP_Extensions.md).


  * `kata-analyze, kata-genmove_analyze [player (optional)] [interval (optional)] ...`
      * Subset of ```kata-analyze``` and ```kata-genmove_analyze```. Support the ```info```, ```move```, ```visits```, ```winrate```, ```prior```, ```lcb```, ```order```, ```pv```, ```scoreLead``` labels. More detail to see [KataGo GTP Extensions](https://github.com/lightvector/KataGo/blob/master/docs/GTP_Extensions.md).


  * Optional Keys
      * All analysis commands support the following keys.
      * ```interval <int>```: Output a line every this many centiseconds.
      * ```minmoves <int>```: There is no effect.
      * ```maxmoves <int>```: Output stats for at most N different legal moves (NOTE: Leela Zero does NOT currently support this field);
      * ```avoid PLAYER VERTEX,VERTEX,... UNTILDEPTH```: Prohibit the search from exploring the specified moves for the specified player, until ```UNTILDEPTH``` ply deep in the search.
      * ```allow PLAYER VERTEX,VERTEX,... UNTILDEPTH```: Equivalent to ```avoid``` on all vertices EXCEPT for the specified vertices. Can only be specified once, and cannot be specified at the same time as ```avoid```.
      * ```ownership True```: Output the predicted final ownership of every point on the board.
      * ```movesOwnership True```: Output the predicted final ownership of every point on the board for every individual move.

## Reinforcement Learning

Please see this [section](./bash/README.md).

## Journal

Please see this [section](./JOURNAL.md).

## Features

* Provide high level player strength on 19x19, depending on hardware.
* Support Sabaki and GoGui analysis mode.
* Support handicap game.
* Support variable komi and board size (from 7x7 to 19x19).
* Lock-free SMP MCTS.
* Acceleration by multi-core processor and multi-Nvidia GPU.
* Predict the current side-to-move winrate and draw-rate.
* Predict the current side-to-move score lead and death strings.
* Reuse the sub-tree.
* Chinese rules with positional superko.
* Gumbel AlphZero learning.

## Todo

* Support Windows platform (CUDA version).
* Support NHWC format.
* Support distributed computation.

## Other Linkings

* Go Text Protocol, [https://www.gnu.org/software/gnugo/gnugo_19.html](https://www.gnu.org/software/gnugo/gnugo_19.html)
* Leela Zero, [https://github.com/leela-zero/leela-zero](https://github.com/leela-zero/leela-zero)
* KataGo methods, [https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
* [You Tube](https://www.youtube.com/watch?v=82UclNrXGxg), playing with Pachi.

## License

The code is released under the GPLv3, except for threadpool.h, cppattributes.h, Eigen and Fast Float, which have specific licenses mentioned in those files.

## Contact

cglemon000@gmail.com (Hung-Tse Lin)
