function(generate_cuda_flags_from_archs)
    if(NOT DEFINED CUDA_ARCHITECTURES)
        message(FATAL_ERROR "CUDA_ARCHITECTURES is not defined")
    endif()

    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.2")
        set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}" PARENT_SCOPE)
    else()
        set(GENCODE_FLAGS "")
        foreach(arch IN LISTS CUDA_ARCHITECTURES)
            string(STRIP "${arch}" arch)
            set(sm "sm_${arch}")
            set(compute "compute_${arch}")

            string(APPEND GENCODE_FLAGS
                " -gencode arch=${compute},code=${sm}"
                " -gencode arch=${compute},code=${compute}"
            )
        endforeach()
        set(CMAKE_CUDA_FLAGS "${GENCODE_FLAGS} ${CMAKE_CUDA_FLAGS}" PARENT_SCOPE)
    endif()
endfunction()

if(NOT DEFINED BLAS_BACKEND)
    set(BLAS_BACKEND "BUILT_IN")
elseif(NOT BLAS_BACKEND STREQUAL "EIGEN"
       AND NOT BLAS_BACKEND STREQUAL "OPENBLAS"
       AND NOT BLAS_BACKEND STREQUAL "CUDA"
       AND NOT BLAS_BACKEND STREQUAL "CUDNN")
    set(BLAS_BACKEND "BUILT_IN")
endif()

if(BLAS_BACKEND STREQUAL "CUDA" OR BLAS_BACKEND STREQUAL "CUDNN")
    project(sayuri LANGUAGES CXX CUDA)
    cmake_minimum_required(VERSION 3.15)
else()
    project(sayuri)
    cmake_minimum_required(VERSION 3.10)
endif()

if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColorReset   "${Esc}[m")
    set(ColorBold    "${Esc}[1m")
    set(ColorRed     "${Esc}[31m")
    set(ColorBoldRed "${ColorRed}${ColorBold}")
endif()

# Custom board size
if(SPECIFIC_BOARD_SIZE)
    message(STATUS "Using the specific board size of ${BOARD_SIZE} x ${BOARD_SIZE}.")
    add_definitions(-DMAX_BOARD_SIZE=${BOARD_SIZE})
endif()

# FP16 support
if(DISABLE_FP16)
    message(STATUS "CUDA FP16 support is disabled.")
else()
    message(STATUS "CUDA FP16 support is enabled.")
    add_definitions(-DENABLE_FP16)
endif()

# Gather all source files
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(BENCHMARK_SOURCES_DIR ${SOURCE_DIR}/benchmark)
set(GAME_SOURCES_DIR ${SOURCE_DIR}/game)
set(NEURAL_SOURCES_DIR ${SOURCE_DIR}/neural)
set(PATTERN_SOURCES_DIR ${SOURCE_DIR}/pattern)
set(MCTS_SOURCES_DIR ${SOURCE_DIR}/mcts)
set(SELFPLAY_SOURCES_DIR ${SOURCE_DIR}/selfplay)
set(UTILS_SOURCES_DIR ${SOURCE_DIR}/utils)

set(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(${IncludePath})

file(GLOB BASIC_SOURCES        ${SOURCE_DIR}/*.cc)
file(GLOB BENCHMARK_SOURCES    ${BENCHMARK_SOURCES_DIR}/*.cc)
file(GLOB GAME_SOURCES         ${GAME_SOURCES_DIR}/*.cc)
file(GLOB NEURAL_SOURCES       ${NEURAL_SOURCES_DIR}/*.cc ${NEURAL_SOURCES_DIR}/blas/*.cc)
file(GLOB PATTERN_SOURCES      ${PATTERN_SOURCES_DIR}/*.cc)
file(GLOB MCTS_SOURCES         ${MCTS_SOURCES_DIR}/*.cc)
file(GLOB SELFPLAY_SOURCES     ${SELFPLAY_SOURCES_DIR}/*.cc)
file(GLOB UTILS_SOURCES        ${UTILS_SOURCES_DIR}/*.cc)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

# Fast parser option
if(USE_FAST_PARSER)
    message(STATUS "Fast parser library enabled.")
    add_definitions(-DUSE_FAST_PARSER)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/fast_float/include/fast_float)
endif()

# AVX instruction set
if(USE_AVX)
    message(STATUS "AVX instructions enabled.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma")
endif()

# zlib support
if(USE_ZLIB)
    message(STATUS "zlib support enabled.")
    add_definitions(-DUSE_ZLIB)
    find_package(ZLIB REQUIRED)
    include_directories(${ZLIB_INCLUDE_DIRS})
endif()

if(BLAS_BACKEND STREQUAL "CUDA" OR BLAS_BACKEND STREQUAL "CUDNN")
    if(BLAS_BACKEND STREQUAL "CUDNN")
        message(STATUS "Backend is set to cuDNN.")
    else()
        message(STATUS "Backend is set to CUDA-only.")
    endif()

    add_definitions(-DUSE_CUDA)
    enable_language(CUDA)
    aux_source_directory(${NEURAL_SOURCES_DIR}/cuda CUDA_SOURCES)

    set(CUDA_STANDARD 14)

    if(${CMAKE_VERSION} VERSION_LESS "3.18.2")
        message(WARNING "We recommend using CMake version 3.18.2 or higher to support the CMAKE_CUDA_ARCHITECTURES feature.")
    endif()

    # https://en.wikipedia.org/wiki/CUDA#GPUs_supported
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
        set(CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 90 120)
        set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets ${CMAKE_CUDA_FLAGS}")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
        set(CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 90)
        set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets ${CMAKE_CUDA_FLAGS}")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        set(CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86 87 90)
        set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets ${CMAKE_CUDA_FLAGS}")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.5)
        set(CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86 87)
        set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets ${CMAKE_CUDA_FLAGS}")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
        set(CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86)
        set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets ${CMAKE_CUDA_FLAGS}")
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
        set(CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80)
        set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets ${CMAKE_CUDA_FLAGS}")
    elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10.2)
        set(CUDA_ARCHITECTURES 30 35 37 50 52 53 60 61 62 70 72 75)
        set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets ${CMAKE_CUDA_FLAGS}")
    else()
        message(WARNING "CUDA 10.2 or greater is recommended, but attempting to build anyways")
        set(CUDA_ARCHITECTURES 30 37 53 70)
    endif()
    generate_cuda_flags_from_archs()
    set(CMAKE_CUDA_FLAGS "-Xptxas -O3 -Xcompiler -fPIC ${CMAKE_CUDA_FLAGS}")

    if(CMAKE_CUDA_FLAGS)
        list(REMOVE_ITEM CMAKE_CUDA_FLAGS "-cudart static")
    endif()
    string(APPEND CMAKE_CUDA_FLAGS "-cudart shared")
    set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
else()
    if(BLAS_BACKEND STREQUAL "EIGEN")
        message(STATUS "Backend is set to Eigen")
    elseif(BLAS_BACKEND STREQUAL "OPENBLAS")
        message(STATUS "Backend is OpenBlas")
    elseif(BLAS_BACKEND STREQUAL "BUILT_IN")
        message(STATUS "Backend is built-in matrix.")
    endif()
endif()

find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(GCC_COMPILE_FLAGS "-Wall -Wextra -ffast-math -march=native -flto")
    set(GCC_DISABLED_WARNING_COMPILE_FLAGS "-Wno-ignored-attributes -Wno-uninitialized -Wno-mismatched-tags")
    set(CMAKE_CXX_FLAGS "${GCC_COMPILE_FLAGS} ${GCC_DISABLED_WARNING_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -Og ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -DNDEBUG ${CMAKE_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "-flto -g")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(GCC_COMPILE_FLAGS "-Wall -Wextra -ffast-math -march=native -flto")
    set(GCC_DISABLED_WARNING_COMPILE_FLAGS "-Wno-ignored-attributes -Wno-maybe-uninitialized -Wno-mismatched-tags")
    set(CMAKE_CXX_FLAGS "${GCC_COMPILE_FLAGS} ${GCC_DISABLED_WARNING_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -Og ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -DNDEBUG ${CMAKE_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "-flto -g")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # pass  
endif()

if(BLAS_BACKEND STREQUAL "EIGEN")
    message(STATUS "Include Eigen blas library")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/Eigen)
    add_definitions(-DUSE_EIGEN)
    add_definitions(-DUSE_BLAS)
elseif(BLAS_BACKEND STREQUAL "OPENBLAS")
    message(STATUS "Include Open BLAS library")
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
endif()

add_executable(sayuri
    ${BASIC_SOURCES}
    ${BENCHMARK_SOURCES}
    ${GAME_SOURCES}
    ${PATTERN_SOURCES}
    ${MCTS_SOURCES}
    ${NEURAL_SOURCES}
    ${UTILS_SOURCES}
    ${SELFPLAY_SOURCES}
    ${CUDA_SOURCES}
    )

target_link_libraries(sayuri Threads::Threads)
target_link_libraries(sayuri ${BLAS_LIBRARIES})
if(USE_ZLIB)
    target_link_libraries(sayuri ${ZLIB_LIBRARIES})
endif()

if(BLAS_BACKEND STREQUAL "CUDA" OR BLAS_BACKEND STREQUAL "CUDNN")
    target_compile_definitions(sayuri PRIVATE USE_CUDA_BACKEND)
    find_package(CUDA REQUIRED)
    if(BLAS_BACKEND STREQUAL "CUDNN")
        find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES cuda/include include)
        if(NOT CUDNN_INCLUDE_DIR)
            message(FATAL_ERROR "${ColorBoldRed} cudnn.h was NOT found, specify CUDNN_INCLUDE_DIR to indicate where it is.")
        endif()
        add_definitions(-DUSE_CUDNN)
        find_library(CUDNN_LIBRARY libcudnn.so PATHS /usr/local/cuda/lib64 /opt/cuda/lib64)
        include_directories(SYSTEM ${CUDNN_INCLUDE_DIR})
    endif()
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    target_link_libraries(sayuri ${CUDNN_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
endif()
