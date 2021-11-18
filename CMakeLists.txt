cmake_minimum_required(VERSION 3.9)
project(Sayuri)

# Set up the root.
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(GAME_SOURCES_DIR ${SOURCE_DIR}/game)
set(NEURAL_SOURCES_DIR ${SOURCE_DIR}/neural)
set(DATA_SOURCES_DIR ${SOURCE_DIR}/data)
set(MCTS_SOURCES_DIR ${SOURCE_DIR}/mcts)
set(UTILS_SOURCES_DIR ${SOURCE_DIR}/utils)

set(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(${IncludePath})

# Set all source file.
set(BASIC_SOURCES
    ${SOURCE_DIR}/main.cc
    ${SOURCE_DIR}/config.cc
    )

set(GAME_SOURCES
    ${GAME_SOURCES_DIR}/board.cc
    ${GAME_SOURCES_DIR}/simple_board.cc
    ${GAME_SOURCES_DIR}/game_state.cc
    ${GAME_SOURCES_DIR}/string.cc
    ${GAME_SOURCES_DIR}/sgf.cc
    ${GAME_SOURCES_DIR}/zobrist.cc
    ${GAME_SOURCES_DIR}/symmetry.cc
    ${GAME_SOURCES_DIR}/gtp.cc
    )

set(NEURAL_SOURCES
    ${NEURAL_SOURCES_DIR}/loader.cc
    ${NEURAL_SOURCES_DIR}/description.cc
    ${NEURAL_SOURCES_DIR}/encoder.cc
    ${NEURAL_SOURCES_DIR}/network.cc
    ${NEURAL_SOURCES_DIR}/fast_policy.cc

    ${NEURAL_SOURCES_DIR}/blas/sgemm.cc
    ${NEURAL_SOURCES_DIR}/blas/blas.cc
    ${NEURAL_SOURCES_DIR}/blas/convolution.cc
    ${NEURAL_SOURCES_DIR}/blas/winograd_convolution3.cc
    ${NEURAL_SOURCES_DIR}/blas/fullyconnect.cc
    ${NEURAL_SOURCES_DIR}/blas/batchnorm.cc
    ${NEURAL_SOURCES_DIR}/blas/biases.cc
    ${NEURAL_SOURCES_DIR}/blas/se_unit.cc
    ${NEURAL_SOURCES_DIR}/blas/blas_forward_pipe.cc
    )

set(DATA_SOURCES
    ${DATA_SOURCES_DIR}/supervised.cc
    ${DATA_SOURCES_DIR}/training.cc
    )

set(MCTS_SOURCES
    ${MCTS_SOURCES_DIR}/time_control.cc
    ${MCTS_SOURCES_DIR}/node.cc
    ${MCTS_SOURCES_DIR}/search.cc
    )

set(UTILS_SOURCES
    ${UTILS_SOURCES_DIR}/log.cc
    ${UTILS_SOURCES_DIR}/parser.cc
    ${UTILS_SOURCES_DIR}/random.cc
    ${UTILS_SOURCES_DIR}/time.cc
    ${UTILS_SOURCES_DIR}/filesystem.cc
    ${UTILS_SOURCES_DIR}/option.cc
    ${UTILS_SOURCES_DIR}/komi.cc
    )

# Prepare basic data.
set(_USE_CUDNN False)
set(_USE_CUDA False)
set(_USE_BLAS False)
set(_USE_EIGEN False)
set(_USE_BUILD_IN False)

if(DEBUG_MODE)
   message(STATUS "Set Debug Mode")
   set(CMAKE_BUILD_TYPE DEBUG)
else()
   set(CMAKE_BUILD_TYPE RELEASE)
endif()

if(USE_FAST_PARSER)
    message(STATUS "Using fast parser library.")
    add_definitions(-DUSE_FAST_PARSER)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/fast_float/include/fast_float)
endif()

if (USE_AVX)
    message(STATUS "Using the AVX instructions")
    set(CMAKE_CXX_FLAGS "-mavx -mfma ${CMAKE_CXX_FLAGS}")
endif()

function(ResetAll)
    set(_USE_CUDNN False)
    set(_USE_CUDA False)
    set(_USE_BLAS False)
    set(_USE_EIGEN False)
    set(_USE_BUILD_IN False)
endfunction()

if(BLAS_BACKEND STREQUAL "EIGEN")
    ResetAll()
    set(_USE_EIGEN True)
elseif(BLAS_BACKEND STREQUAL "OPENBLAS")
    ResetAll()
    set(_USE_BLAS True)
elseif(BLAS_BACKEND STREQUAL "CUDA")
    ResetAll()
    set(_USE_CUDA True)
elseif(BLAS_BACKEND STREQUAL "CUDNN")
    ResetAll()
    set(_USE_CUDA True)
    set(_USE_CUDNN True)
else()
    ResetAll()
    set(_USE_BUILD_IN True)
endif()

# Define the neural network default backend.
if(_USE_CUDA)
    if(_USE_CUDNN)
        message(STATUS "Using cuDNN backend.")
    else()
        message(STATUS "Using CUDA backend.")
        message(" The cuDNN backend is faster than CUDA backend in multi-batch condition.")
        message(" If you want to use cuDNN to speed up, adding flag -DBLAS_BACKEND=CUDNN.")
        message(" The cuDNN library and Nvida GPU are both required.\n")
    endif()

    add_definitions(-DUSE_CUDA)
    enable_language(CUDA)
    aux_source_directory(${NEURAL_SOURCES_DIR}/cuda CUDA_SOURCES)
    set(CMAKE_CUDA_FLAGS "" CACHE STRING "")
    if(CMAKE_CUDA_FLAGS)
        list(REMOVE_ITEM CMAKE_CUDA_FLAGS "-cudart static")
    endif()
    string(APPEND CMAKE_CUDA_FLAGS "-cudart shared")

    set(CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS
        "-gencode arch=compute_30,code=sm_30 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 -gencode  arch=compute_75,code=compute_75"
    )
    set(CMAKE_CUDA_FLAGS "-Xptxas -O3 -Xcompiler -fPIC ${CMAKE_CUDA_FLAGS}")
else()
    if(_USE_EIGEN)
        message(STATUS "Use Eigen backend")
    elseif(_USE_BLAS)
        message(STATUS "Use OpenBlas backend")
    elseif(_USE_BUILD_IN)
        message(STATUS "Using built-in matrix backend.")
        message(" Built-in matrix is much slow. Open Blas and Eigen are the better choice.")
        message(" If you want to use Eigen, adding flag -DBLAS_BACKEND=EIGEN. And you need")
        message(" to put the Eigen library to third_party directory. If you want to use OpenBlas,")
        message(" adding flag -DBLAS_BACKEND=OPENBLAS. OpenBlas library is required.\n")
    endif()

    message(" The GPU backend is much faster than CPU backend.")
    message(" If you want to use GPU to speed up, adding flag -DBLAS_BACKEND=CUDA or -DBLAS_BACKEND=CUDNN.")
    message(" CUDA library and Nvida GPU both are required.\n")
endif()

# Find all required packages.
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(GCC_COMPILE_FLAGS "-Wall -Wextra -ffast-math -march=native -flto")
set(GCC_DISABLED_WARNING_COMPILE_FLAGS "-Wno-ignored-attributes -Wno-maybe-uninitialized -Wno-mismatched-tags")
set(CMAKE_CXX_FLAGS "${GCC_COMPILE_FLAGS} ${GCC_DISABLED_WARNING_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -DNDEBUG ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "-flto -g")

if(_USE_EIGEN)
    message(STATUS "Include Eigen blas library")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/Eigen)
    add_definitions(-DUSE_EIGEN)
    add_definitions(-DUSE_BLAS)
elseif(_USE_BLAS)
    message(STATUS "Include Open BLAS library")
    message(" Looking for system BLAS/OpenBLAS library...")
    find_package(BLAS REQUIRED)
    find_path(
        BLAS_INCLUDE_DIRS openblas_config.h
        /usr/include
        /usr/local/include
        /usr/include/openblas
        /opt/OpenBLAS/include
        /usr/include/x86_64-linux-gnu
        $ENV{BLAS_HOME}/include)
    add_definitions(-DUSE_OPENBLAS)
    add_definitions(-DUSE_BLAS)
    if((UNIX AND NOT APPLE) OR WIN32)
        include_directories(${BLAS_INCLUDE_DIRS})
    endif()
    if(APPLE)
        include_directories("/System/Library/Frameworks/Accelerate.framework/Versions/Current/Headers")
    endif()
    message(" BLAS/OpenBLAS library be found.\n")
endif()

add_executable(Sayuri
    ${BASIC_SOURCES}
    ${GAME_SOURCES}
    ${DATA_SOURCES}
    ${MCTS_SOURCES}
    ${NEURAL_SOURCES}
    ${UTILS_SOURCES}
    ${CUDA_SOURCES}
    )

target_link_libraries(Sayuri Threads::Threads)
target_link_libraries(Sayuri ${BLAS_LIBRARIES})

if(_USE_CUDA)
    target_compile_definitions(Sayuri PRIVATE USE_CUDA_BACKEND)
    find_package(CUDA REQUIRED)
    if(_USE_CUDNN)
        message(STATUS "Include CuDNN library")
        message(" Looking for cuDNN library...")
        find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES cuda/include include)
        if((NOT CUDNN_INCLUDE_DIR))
            message(ERROR "${ColorBoldRed} cudnn.h was NOT found, specify CUDNN_INCLUDE_DIR to indicate where it is.")
        endif()
        add_definitions(-DUSE_CUDNN)
        find_library(CUDNN_LIBRARY libcudnn.so PATHS /usr/local/cuda/lib64 /opt/cuda/lib64)
        include_directories(SYSTEM ${CUDNN_INCLUDE_DIR})
        message(" The cuDNN library be found.\n")
    endif()
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    target_link_libraries(Sayuri ${CUDNN_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
endif()

