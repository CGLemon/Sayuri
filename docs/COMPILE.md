## Compile

Document for building the program from source.

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

Accelerate the network forwarding pipe by GPUs. CUDA is required.

    $ cmake .. -DBLAS_BACKEND=CUDA

Accelerate the network forwarding pipe by GPUs. CUDA and cuDNN are both required. This backend is more steady than CUDA-only backend.

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
        $ g++ -std=c++14 -ffast-math -I . -lpthread *.cc utils/*.cc summary/*.cc game/*.cc mcts/*.cc neural/*.cc neural/blas/*.cc neural/cuda/*.cc pattern/*.cc selfplay/*.cc -o Sayuri -O3 -DNDEBUG -DWIN32 -I ../third_party/Eigen -DUSE_BLAS -DUSE_EIGEN


