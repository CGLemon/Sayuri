
<div id="sayuri-art" align="center">
    <br/>
    <img src="./img/sayuri-art.PNG" alt="Sayuri Art" width="768"/>
    <h3>Sayuri</h3>
</div>

## Let's ROCK!

Sayuri is a GTP-compliant go engine based on Deep Convolutional Neural Network and Monte Carlo Tree Search. She is strongly inspired by Leela Zero and Kata Go. The board data structure, search algorithm and network format are borrowed from Leela Zero in the beginning. Current version follow the Kata Go research, the engine supports variable komi and board size now. Some methods you may see my Hackmd article (chinese).

* [開發日誌](https://hackmd.io/@yrHb-fKBRoyrKDEKdPSDWg/BJgfay0Yc)

## Requirements

* Ubuntu or MacOS only
* GCC, Clang, must support C++14 or higher
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

Accelerate the network fowardind pipe by CPU. OpenBlas or Eigen are required. OpenBlas and Eigen are significantly faster than built-in blas. OpenBlas is recommended on MacOS.

    $ cmake .. -DBLAS_BACKEND=OPENBLAS

or

    $ cmake .. -DBLAS_BACKEND=EIGEN

Accelerate the network fowardind pipe by GPUs. CUDA is required.

    $ cmake .. -DBLAS_BACKEND=CUDA

Accelerate the network fowardind pipe by GPUs. CUDA and cuDNN are required. It will be faster than CUDA-only in the most cases.

    $ cmake .. -DBLAS_BACKEND=CUDNN


Accelerate to load the network file. Fast Float library is required.

    $ cmake .. -USE_FAST_PARSER=1


## Weights and Book

You may download the weights file and opening book from my [google drive](https://drive.google.com/drive/folders/1OiVcIwewcIh5nnmR8pBFKMSdkbBYNF2c?usp=sharing). The current weights size is 15 blocks and 192 filters. The opening book is human-like book, trained on profession games. Force Sayuri to play variable opening moves. It is just fun for playing.

* The renorm prefix means applying the batch renormalization, and the fixup means fixup initialization.

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
|  --no-dcnn              | None   | Experimental, disable network, very weak.      |
|  --help, -h             | None   | Show the more arguments.                       |
    

Default setting: will select reasonable thread and batch size, 10 seconds per move, all GPU devices

    $ ./Sayuri -w <weights file>

Example setting 1: 4 thread, 2 batches and 12800 playouts
    
    $ ./Sayuri -w <weights file> -t 4 -b 2 -p 12800

Example setting 2: quickly and friendly pass game
    
    $ ./Sayuri -w <weights file> -t 1 -b 1 --const-time 1 --friendly-pass --reuse-tree

Example setting 3: directly policy output 

    $ ./Sayuri -w <weights file> -t 1 -b 1 -p 1

Example setting 4: use the GPU 0 and GPU 2

    $ ./Sayuri -w <weights file> --gpu 0 --gpu 2

Experimental setting: disable network forwarding pipe, very weak

    $ ./Sayuri -w <weights file> --no-dcnn

## Generate Opening Book

You need to collect enough SGF games (at least over 10000 games). Then, go to the GTP mode and enter follow command. Wait some time to generate a new opening book.

    genbook <SGF file> <output name>

## Graphical Interface

Sayuri is not complete engine. You need a graphical interface for playing with her. She supports any GTP (version 2) interface application. [Sabaki](https://sabaki.yichuanshen.de/) and [GoGui](https://github.com/Remi-Coulom/gogui) are recommended that because Sayuri support some specific analysis commands. 

* Sabaki analysis mode

![sabaki-sample01](./img/sabaki-sample01.png)

* GoGui analysis commands

![gogui-sample01](./img/gogui-sample01.png)

## Analysis Commands

The engine supports the following GTP analysis commands.

  * `analyze, genmove_analyze [player (optional)] [interval (optional)] ...`
      * The behavior is same as lz-analyze, lz-genmove_analyze 

  * `lz-analyze, lz-genmove_analyze [player (optional)] [interval (optional)] ...`
      * Extension GTP commands of lz-analyze and lz-genmove_analyze. Support the ```info```, ```move```, ```visits```, ```winrate```, ```prior```, ```lcb```, ```order```, ```pv```,```scoreLead``` labels. More detail to see [KataGo GTP Extensions](https://github.com/lightvector/KataGo/blob/master/docs/GTP_Extensions.md).


  * `kata-analyze, kata-genmove_analyze [player (optional)] [interval (optional)] ...`
      * Subset of kata-analyze and kata-genmove_analyze. Support the ```info```, ```move```, ```visits```, ```winrate```, ```prior```, ```lcb```, ```order```, ```pv```,```scoreLead``` labels. More detail to see [KataGo GTP Extensions](https://github.com/lightvector/KataGo/blob/master/docs/GTP_Extensions.md).


  * Optional Keys
      * All analysis commands support the following keys.
      * ```interval <int>```: Output a line every this many centiseconds. 
      * ```ownership True```: Output the predicted final ownership of every point on the board.
      * ```movesOwnership True```: Output the predicted final ownership of every point on the board for every individual move.

## Misc

### About this engine

The project was began from the Aug 6, 2019. In the beginning, I just wanted to write a Go Bot that could beat lower level player in the 9x9 board. Although It was easy to train a strong enough bot with deep learning technique, it was hard for me to do that in that time. It is because that I do not major in computer science and I never learn the C++ then before. After few years learning, my C++ skill is much better. Even more, the current version can beat me in any board size.

### About the ancient technique

Before the AlphaGo (2016s), the most state-of-the-art computer go combine the MCTS and MM (Minorization-Maximization). Crazy Stone and Zen use that. Or combining the MCTS and SB (Simulation Balancing). The Eric (predecessor of AlphaGo) and Leela use that. Ray, one of the strongest open source go engine before AlphaGo, writed by Yuki Kobayashi which based on the MM algorithm. I am surprised that it can play the game well wihout much human knowledge and Neural Network. What's more, it can beats high level go player on 9x9 if we provide it enough computation. But thanks for deep learning technique, the computer go engine is significantly stronger than before. Sayuri can beat the Ray on 19x19 with only policy network. This result show the advantage of deep Neural Network.

Although the Neural Network base engines are more powerful, you may still try some engine with non Neural Network and feel the power of ancient technique. Here is the list.

* [Leela](https://www.sjeng.org/leela.html), need to add the option ```--nonets``` to disable DCNN.
* [Pachi](https://github.com/pasky/pachi), need to add the option ```--nodcnn``` to disable DCNN.
* [Ray](https://github.com/kobanium/Ray), may be strongest open source engine in the 2016s.

I am trying to implement this ancient technique currently. Merge the MM patterns base and the DCNN base technique to provide widely dynamic strength. It should be fun.

## Features

* Support Sabaki and GoGui analysis mode.
* Support handicap game.
* Support variable komi and board size (from 7x7 to 19x19).
* Lock-free SMP MCTS.
* Acceleration by multi-core processor and multi-Nvidia GPU.
* Predict the current side winrate and draw-rate.
* Predict the current side score lead and death strings.
* Reuse the sub-tree.
* Chinese rule.

## Todo

* Sopport Windows platform.
* Support half-float.
* Support NHWC format.
* Support distributed computation.
* Improve the non-DCNN mode strength.

## Other Linkings

* Go Text Protocol, [https://www.gnu.org/software/gnugo/gnugo_19.html](https://www.gnu.org/software/gnugo/gnugo_19.html)
* Leela Zero, [https://github.com/leela-zero/leela-zero](https://github.com/leela-zero/leela-zero)
* Kata Go methods, [https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
* [You Tube](https://www.youtube.com/watch?v=82UclNrXGxg), playing with Pachi.

## License

The code is released under the GPLv3, except for threadpool.h, cppattributes.h, Eigen and Fast Float, which have specific licenses mentioned in those files.

## Contact

cglemon000@gmail.com (Hung-Zhe Lin)
