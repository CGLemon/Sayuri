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

## Windows Version (Experimental)

Only support for compiling the executable file via command line now. Assume you already donwload the source code and Eigen. Then, you need to download the Visual Studio 2022/2019 and install depending C++ library.  Here are two different compilers. Note that we must use ```x64 Native Tools Command Prompt for VS XXX``` environment or power shell.

* Via [MinGW](https://github.com/mstorsjo/llvm-mingw),  CPU-only version

        $ g++ -std=c++14 -ffast-math -I . -lpthread *.cc utils/*.cc game/*.cc mcts/*.cc neural/*.cc neural/blas/*.cc neural/cuda/*.cc pattern/*.cc selfplay/*.cc -o sayuri -O3 -DNDEBUG -DWIN32 -I ../third_party/Eigen -DUSE_BLAS -DUSE_EIGEN -static

* Via NVCC, CPU-only and GPU version

        // CPU-only version
        $ nvcc main.cc config.cc version.cc game/board.cc game/book.cc game/game_state.cc game/gtp.cc game/iterator.cc game/pattern_board.cc game/sgf.cc game/strings.cc game/symmetry.cc game/zobrist.cc mcts/node.cc mcts/search.cc mcts/time_control.cc neural/description.cc neural/encoder.cc neural/loader.cc neural/network.cc neural/training_data.cc neural/winograd_helper.cc neural/blas/batchnorm.cc neural/blas/biases.cc neural/blas/blas.cc neural/blas/blas_forward_pipe.cc neural/blas/convolution.cc neural/blas/fullyconnect.cc neural/blas/se_unit.cc neural/blas/sgemm.cc neural/blas/winograd_convolution3.cc neural/cuda/cuda_common.cc neural/cuda/cuda_forward_pipe.cc neural/cuda/cuda_layers.cc neural/cuda/cuda_kernels.cu pattern/gammas_dict.cc pattern/mm.cc pattern/mm_trainer.cc pattern/pattern.cc selfplay/engine.cc selfplay/pipe.cc utils/filesystem.cc utils/gogui_helper.cc utils/gzip_helper.cc utils/komi.cc utils/log.cc utils/option.cc utils/parse_float.cc utils/random.cc utils/splitter.cc utils/time.cc -o sayuri  -I . -DNDEBUG -DWIN32 -DNOMINMAX  -I ../third_party/Eigen -DUSE_BLAS -DUSE_EIGEN -O3 -Xcompiler /O2 -Xcompiler /std:c++14
        
        // GPU version
        $ nvcc main.cc config.cc version.cc game/board.cc game/book.cc game/game_state.cc game/gtp.cc game/iterator.cc game/pattern_board.cc game/sgf.cc game/strings.cc game/symmetry.cc game/zobrist.cc mcts/node.cc mcts/search.cc mcts/time_control.cc neural/description.cc neural/encoder.cc neural/loader.cc neural/network.cc neural/training_data.cc neural/winograd_helper.cc neural/blas/batchnorm.cc neural/blas/biases.cc neural/blas/blas.cc neural/blas/blas_forward_pipe.cc neural/blas/convolution.cc neural/blas/fullyconnect.cc neural/blas/se_unit.cc neural/blas/sgemm.cc neural/blas/winograd_convolution3.cc neural/cuda/cuda_common.cc neural/cuda/cuda_forward_pipe.cc neural/cuda/cuda_layers.cc neural/cuda/cuda_kernels.cu pattern/gammas_dict.cc pattern/mm.cc pattern/mm_trainer.cc pattern/pattern.cc selfplay/engine.cc selfplay/pipe.cc utils/filesystem.cc utils/gogui_helper.cc utils/gzip_helper.cc utils/komi.cc utils/log.cc utils/option.cc utils/parse_float.cc utils/random.cc utils/splitter.cc utils/time.cc -o sayuri  -I . -DNDEBUG -DWIN32 -DNOMINMAX -DUSE_CUDA -lcudart -lcublas -O3 -Xcompiler /O2 -Xcompiler /std:c++14

