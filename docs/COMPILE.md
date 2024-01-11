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

Accelerate the network forwarding pipe by CPU. OpenBLAS are required. Eigen library is included in the ```third_party``` directory. OpenBLAS and Eigen are significantly faster than built-in blas.

    $ cmake .. -DBLAS_BACKEND=OPENBLAS

or

    $ cmake .. -DBLAS_BACKEND=EIGEN

Accelerate the network forwarding pipe by GPUs. CUDA is required. This backend is fastest in most case.

    $ cmake .. -DBLAS_BACKEND=CUDA

Accelerate the network forwarding pipe by GPUs. CUDA and cuDNN are both required. This backend is more steady than CUDA-only backend.

    $ cmake .. -DBLAS_BACKEND=CUDNN

Compile a bigger board size version. Set it as 0 to disable this option. Only support for below 25x25 size.

    $ cmake .. -DBOARD_SIZE=25

Disable the FP16 supported if your CUDA version doesn't support for it.

    $ cmake .. -DDISABLE_FP16=1

Compress the training data file. It can save memory usage during the self-play process.

    $ cmake .. -DUSE_ZLIB=1

## Windows Version (Experiment)

The Windwos version is still in progress. The performance of GPU version on Windows is slower than Linux. But it as least work well on the Windows 10/11.

1. Download the Visual Studio (Only test the 2022 verion).
2. Download the MinGW from [here](https://github.com/mstorsjo/llvm-mingw).
3. Clone the GitHub repo and compile it.

        $ git clone https://github.com/CGLemon/Sayuri
        $ cd Sayuri
        $ git submodule update --init --recursive
        $ cd src
        $ g++ -std=c++14 -ffast-math -I . -lpthread *.cc utils/*.cc summary/*.cc game/*.cc mcts/*.cc neural/*.cc neural/blas/*.cc neural/cuda/*.cc pattern/*.cc selfplay/*.cc -o sayuri -O3 -DNDEBUG -DWIN32 -I ../third_party/Eigen -DUSE_BLAS -DUSE_EIGEN -static

If you want to compile the CUDA-only version, you need to download the CUDA toolkit, such CUDA 12. Then use the NVCC compiler instead of GCC.

    $ nvcc main.cc config.cc version.cc game/board.cc game/book.cc game/game_state.cc game/gtp.cc game/iterator.cc game/pattern_board.cc game/sgf.cc game/strings.cc game/symmetry.cc game/zobrist.cc mcts/node.cc mcts/search.cc mcts/time_control.cc neural/description.cc neural/encoder.cc neural/loader.cc neural/network.cc neural/training.cc neural/winograd_helper.cc neural/blas/batchnorm.cc neural/blas/biases.cc neural/blas/blas.cc neural/blas/blas_forward_pipe.cc neural/blas/convolution.cc neural/blas/fullyconnect.cc neural/blas/se_unit.cc neural/blas/sgemm.cc neural/blas/winograd_convolution3.cc neural/cuda/cuda_common.cc neural/cuda/cuda_forward_pipe.cc neural/cuda/cuda_layers.cc neural/cuda/cuda_kernels.cu pattern/gammas_dict.cc pattern/mm.cc pattern/mm_trainer.cc pattern/pattern.cc selfplay/engine.cc selfplay/pipe.cc summary/accuracy.cc summary/selfplay_accumulation.cc utils/filesystem.cc utils/gogui_helper.cc utils/gzip_helper.cc utils/komi.cc utils/log.cc utils/option.cc utils/parse_float.cc utils/random.cc utils/splitter.cc utils/time.cc -o sayuri  -I . -DNDEBUG -DWIN32 -DNOMINMAX -DUSE_CUDA -lcudart -lcublas -O3 -Xcompiler /O2 -Xcompiler /std:c++14



