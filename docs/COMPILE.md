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

## CMake Options

We offer CMake for compilation on platforms like Linux and macOS, with support for the following options:

### Accelerate with CPU

You can accelerate the network forwarding pipeline using your CPU. OpenBLAS and Eigen are required for this. Both libraries are significantly faster than built-in BLAS implementations. The Eigen library should be included in the ```third_party``` directory.

To use OpenBLAS:

    $ cmake .. -DBLAS_BACKEND=OPENBLAS

To use Eigen:

    $ cmake .. -DBLAS_BACKEND=EIGEN


### Accelerate with GPUs

To accelerate the network forwarding pipeline using GPUs, CUDA is required. This backend is typically the fastest option.

    $ cmake .. -DBLAS_BACKEND=CUDA

For a more stable experience with GPU acceleration, both CUDA and cuDNN are required.

    $ cmake .. -DBLAS_BACKEND=CUDNN

### Compile with a Larger Board Size

You can compile a version that supports a larger board size. Set this option to 0 to disable it. This feature currently only supports board sizes up to 25x25.

    $ cmake .. -DSPECIFIC_BOARD_SIZE=25

### Disable FP16 Support

If your CUDA version does not support FP16, you can disable it during compilation.

    $ cmake .. -DDISABLE_FP16=1

### Compress Training Data

To save memory usage during the self-play process, you can compress training data files.

    $ cmake .. -DUSE_ZLIB=1

## Windows

To compile the executable, we provide a ```.bat``` file that supports both CPU and GPU versions. For the CPU version, we default to using Eigen as the backend. For the GPU version, CUDA is used as the backen

Before you begin, you must first download and install Visual Studio 2022/2019 along with the necessary C++ libraries. After installation, execute the following commands. It's crucial to run these commands from the ```x64 Native Tools Command Prompt for VS XXX``` environment or PowerShell.

### CPU Version

This version requires the GCC compiler. You can use MinGW for this. To compile the CPU version, enter:

    .\build.bat gcc


### GPU Version

This version requires the NVCC compiler. To compile the GPU version, enter:

    .\build.bat nvcc
