
<div id="sayuri-art" align="center">
    <br/>
    <img src="./img/sayuri-art.PNG" alt="Sayuri Art" width="768"/>
    <h3>Sayuri</h3>
</div>

## Let's ROCK!

Sayuri is a GTP-compliant go engine based on Deep Convolutional Neural Network and Monte Carlo Tree Search. Learning the game of Go from scratch without strategic knowledge from human with AlphaZero-based algorithm. She is strongly inspired by Leela Zero and KataGo. The board data structure, search algorithm and network format are borrowed from Leela Zero in the beginning. Current version follows the KataGo research, the engine supports variable komi and board size now. Some methods or reports you may see my HackMD articles (some are chinese).

* [開發日誌](https://hackmd.io/@yrHb-fKBRoyrKDEKdPSDWg/BJgfay0Yc)
* [AlphaZero 之加速演算法實作](https://hackmd.io/@yrHb-fKBRoyrKDEKdPSDWg/HJI9_p70i)
* [The performance report before UEC15](https://drive.google.com/file/d/1ATd_u-E-OnviczsDH8wVL0c3Q1NzUCKW/view?usp=share_link)
* [Journal](./docs/JOURNAL.md).

## Quick Start via Terminal

First, you need a executable weights. Download the last v0.6 weights [here](https://drive.google.com/drive/folders/1nawHAKHTBKEpLcizaVrK4GVDSIuVqJ-Q?usp=sharing) and see the current RL progression [here](https://hackmd.io/@yrHb-fKBRoyrKDEKdPSDWg/HJew5OFci). If you want to use the old network, please use the v0.5 engine.

Then start the program with GTP mode via the terminal/PowerShell, please enter

    $ ./Sayuri -w <weights file> -t 1 -b 1 -p 400

You will see the diagnostic verbose. If the verbose includes ```Network Verison``` information, it means you success to start the program. For more arguments, please give the ```--help``` option.

## Graphical Interface

Sayuri is not complete engine. You need a graphical interface for playing with her. She supports any GTP (version 2) interface application. [Sabaki](https://sabaki.yichuanshen.de/) and [GoGui](https://github.com/Remi-Coulom/gogui) are recommended that because Sayuri supports some specific analysis commands. 

* Sabaki analysis mode

![sabaki-sample01](./img/sabaki-sample01.png)

* GoGui analysis commands

![gogui-sample01](./img/gogui-sample01.png)

## Build From Souce

Please see this [section](./docs/COMPILE.md).

## Reinforcement Learning

Please see this [section](./bash/README.md).

## Todo

* Support Windows platform (CUDA version).
* Support NHWC format.
* Support distributed computation.

## Other Linkings

* Go Text Protocol, [https://www.gnu.org/software/gnugo/gnugo_19.html](https://www.gnu.org/software/gnugo/gnugo_19.html)
* Leela Zero, [https://github.com/leela-zero/leela-zero](https://github.com/leela-zero/leela-zero)
* KataGo methods, [https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
* [You Tube](https://www.youtube.com/watch?v=82UclNrXGxg), playing with Pachi.
* Supported analysis commands, [analyze](./docs/ANALYZE.md).

## License

The code is released under the GPLv3, except for threadpool.h, cppattributes.h, Eigen and Fast Float, which have specific licenses mentioned in those files.

## Contact

cglemon000@gmail.com (Hung-Tse Lin)
