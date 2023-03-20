
<div id="sayuri-art" align="center">
    <br/>
    <img src="./img/sayuri-art.PNG" alt="Sayuri Art" width="768"/>
    <h3>Sayuri</h3>
</div>

## Let's ROCK!

Sayuri is a GTP-compliant go engine based on Deep Convolutional Neural Network and Monte Carlo Tree Search. She is strongly inspired by Leela Zero and KataGo. The board data structure, search algorithm and network format are borrowed from Leela Zero in the beginning. Current version follows the KataGo research, the engine supports variable komi and board size now. Some methods you may see my HackMD article (in chinese).

* [開發日誌](https://hackmd.io/@yrHb-fKBRoyrKDEKdPSDWg/BJgfay0Yc)

## Requirements

* Ubuntu or MacOS only
* GCC, Clang, must support C++14 or higher
* CMake 3.15 or later
* Optional: Eigen or OpenBLAS library
* Optional: CUDA 10.x or 11.x library (GCC 7.x passed)
* Optional: cuDNN 7.x or 8.x library
* Optional: gzip

## Default Compiling (Ubuntu or MacOS)

    $ git clone https://github.com/CGLemon/Sayuri
    $ cd Sayuri
    $ git submodule update --init --recursive
    $ mkdir build && cd build
    $ cmake ..
    $ make -j

## Optional Compiling

Accelerate the network forwarding pipe by CPU. OpenBlas or Eigen are required. OpenBlas and Eigen are significantly faster than built-in blas. OpenBlas is recommended on MacOS.

    $ cmake .. -DBLAS_BACKEND=OPENBLAS

or

    $ cmake .. -DBLAS_BACKEND=EIGEN

Accelerate the network forwarding pipe by GPUs. CUDA is required. It is the faster backend.

    $ cmake .. -DBLAS_BACKEND=CUDA

Accelerate the network forwarding pipe by GPUs. CUDA and cuDNN are both required. This backend is much steady but may be slower than CUDA-only.

    $ cmake .. -DBLAS_BACKEND=CUDNN

Compile a bigger board size version. It will use the default size if we set it as 0. 

    $ cmake .. -DBOARD_SIZE=25

Disable the FP16 CUDA code if your CUDA version doesn't support for it.

    $ cmake .. -DDISABLE_FP16=1

Save the compressed training data file. It can save many memory usage in the self-play.

    $ cmake .. -DUSE_ZLIB=1

## Weights and Others.

You may download the SL weights file, opening book and patterns from my [google drive](https://drive.google.com/drive/folders/1cXAoOghgkUfNVZWRzEyvfB4uY_TTbaVH?usp=share_link). Here is the description list. Because I may update the network format or encoder, be sure that you download the correspond weights for the last engine. I do not promise the any file is backward compatible.

| File                    | Description                                    |
| :---------------------- | :--------------------------------------------- |
| Network Weights         | The main weights file, trained on KataGo self-play games. The ```.bin``` postfix is binary version. |
| Opening Book            | It is human-like book, gathering from profession games. Force Sayuri to play variable opening moves. It is just fun for playing. |
| MM Patterns             | It is for no-dcnn mode, trained on the games of high level players from KGS. |

<br/>

Download last RL weights file from [zero](https://drive.google.com/drive/folders/1PlPTOH1amP3J7HR5uxi9Q_Dd_CL9rEX8?usp=share_link) directory. The file name looks like  ```zero-10k.bin.txt```. The ```10k``` means it played 10000 self-play games. The self-play note is [here](https://hackmd.io/@yrHb-fKBRoyrKDEKdPSDWg/HJew5OFci).

## Engine Arguments

Here are some useful arguments which you may need.

| Arguments               | Param  | Description                                    |
| :---------------------- | :----- | :--------------------------------------------- |
|  --weights, -w          | string | File with network weights.                     |
|  --patterns             | string | File with patterns.                            |
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
|  --no-dcnn              | None   | Disable network, very weak.                    |
|  --no-fp16              | None   | Disable FP16 computation.                      |
|  --no-winograd          | None   | Disable WinoGrad algorithm.                    |
|  --help, -h             | None   | Show the more arguments.                       |
    
<br/>

Default setting: will select reasonable threads and batch size, 10 seconds per move, all GPU devices

    $ ./Sayuri -w <weights file>

Example setting 1: 4 threads, 2 batches and 12800 playouts
    
    $ ./Sayuri -w <weights file> -t 4 -b 2 -p 12800

Example setting 2: quickly and friendly pass game
    
    $ ./Sayuri -w <weights file> -t 1 -b 1 --const-time 1 --friendly-pass --reuse-tree

Example setting 3: set 0 playouts, directly policy output 

    $ ./Sayuri -w <weights file> -t 1 -b 1 -p 0

Example setting 4: use the GPU 0 and GPU 2

    $ ./Sayuri -w <weights file> --gpu 0 --gpu 2

Example setting 5: disable the network forwarding pipe, arond 5k on 9x9, 10k on 19x19. The ```--lcb-reduction``` should be set ```1```

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

Please see this [section](./bash/RL.md).

## Misc

### About the ancient technology

(August, 2022)

Before the AlphaGo (2016s), the most of state-of-the-art computer Go combine the MCTS and MM (Minorization-Maximization). Crazy Stone and Zen use that. Or combining the MCTS and SB (Simulation Balancing). The Eric (predecessor of AlphaGo) and Leela use that. Ray, one of the strongest open source Go engine before AlphaGo, writed by Yuki Kobayashi which is based on the MM algorithm. I am surprised that it can play the game well without much human knowledge and Neural Network. What's more, it can beat high level Go player on 9x9 if we provide it enough computation. But thanks for deep learning technique, the computer Go engine is significantly stronger than before. Sayuri can beat the Ray (v10.0) on 19x19 with only policy network. This result shows the advantage of Neural Network technology.

Although the Neural Network based engines are more powerful, I still recommend you to try some engine with non Neural Network and feel the power of ancient technology. Here is the list.

* [Leela](https://www.sjeng.org/leela.html), need to add the option ```--nonets``` to disable DCNN.
* [Pachi](https://github.com/pasky/pachi), need to add the option ```--nodcnn``` to disable DCNN.
* [Ray](https://github.com/kobanium/Ray), may be strongest open source engine before the 2016s.

I am trying to implement this ancient technique. Merge the MM patterns based and the DCNN based technique to provide widely dynamic strength. It should be fun.

### The Gumbel learning

(November, 2022)

On the 2022 CGF Open, the Ray author, Yuki Kobayashi, implemented a new algorithm called Gumbel learning. it is a effective trick for AlphaZero and it guarantees to improve policy with low playouts. As far as I know, Ray is the first successful superhuman level engine with Gumbel learning on 19x19. Inspired by Ray, I decide to implement this ideal in my project. Hope that this project would become another successful Gumbel learning engine.

* [Policy improvement by planning with Gumbel](https://www.deepmind.com/publications/policy-improvement-by-planning-with-gumbel)
* [Ray's apeal letter for UEC 14](https://drive.google.com/file/d/1yLjGboOLMOryhHT-aWG_0zAF-G7LDcTH/view)

### Improve the network performance

(February, 2023)

The Ray author, Yuki Kobayashi, proposed three points which may improve my network performance. Here are list.

* The half floating-point.
* The NHWC format.
* Bottleneck network, It may improve 30% speed without losing accuracy.

KataGo also used the variant bottleneck and said it could significantly improve the performance. This result shows the advance of these kinds of structure.


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

## Todo

* Support Windows platform.
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

cglemon000@gmail.com (Hung-Zhe Lin)
