@echo off
setlocal enabledelayedexpansion

REM Set source directories
set "SRC_DIR=src"
set "BENCHMARK_DIR=%SRC_DIR%\benchmark"
set "GAME_DIR=%SRC_DIR%\game"
set "NEURAL_DIR=%SRC_DIR%\neural"
set "NEURAL_BLAS_DIR=%SRC_DIR%\neural\blas"
set "NEURAL_CUDA_DIR=%SRC_DIR%\neural\cuda"
set "PATTERN_DIR=%SRC_DIR%\pattern"
set "MCTS_DIR=%SRC_DIR%\mcts"
set "SELFPLAY_DIR=%SRC_DIR%\selfplay"
set "UTILS_DIR=%SRC_DIR%\utils"

REM --- Compiler Selection ---
REM Check if a compiler choice is provided as a command-line argument
if "%~1"=="" (
    REM If no argument, default to gcc
    set "COMPILER_CHOICE=gcc"
) else (
    REM Use the provided argument as the compiler choice
    set "COMPILER_CHOICE=%~1"
)

REM Set compilation output
set "OUTPUT=sayuri.exe"

REM Initialize file collection
set "FILES="

REM --- Common Compilation Options ---
REM These options apply to both GCC and NVCC (where applicable)
set "COMMON_OPTIONS=-I %SRC_DIR% -DNDEBUG -DWIN32 -DNOMINMAX -O3"

REM --- Conditional Compiler-Specific Options and File Collection ---
if /I "%COMPILER_CHOICE%"=="nvcc" (
    set "COMPILER_CMD=nvcc"

    REM --- Step 1: Detect CUDA Version ---
    echo Detecting CUDA version...
    set "CUDA_MAJOR=0"
    set "CUDA_MINOR=0"
    set "CUDA_ARCH_FLAGS="
    
    REM This logic is updated to correctly parse the version from the "release" line.
    for /f "tokens=4" %%a in ('nvcc --version 2^>nul ^| findstr "release"') do (
        set "version_part=%%a"
        set "version_part=!version_part:,=!"
        for /f "tokens=1,2 delims=." %%b in ("!version_part!") do (
            set "CUDA_MAJOR=%%b"
            set "CUDA_MINOR=%%c"
        )
    )

    REM Only set architecture flags if a CUDA version was successfully detected.
    if !CUDA_MAJOR! NEQ 0 (
        echo Found CUDA Version: !CUDA_MAJOR!.!CUDA_MINOR!

        REM --- Step 2: Set CUDA Architectures based on version (logic from your CMake) ---
        set "ARCH_LIST="
        if !CUDA_MAJOR! GEQ 12 (
            if !CUDA_MINOR! GEQ 8 (
                set "ARCH_LIST=50 52 53 60 61 62 70 72 75 80 86 87 90 120"
            ) else (
                set "ARCH_LIST=50 52 53 60 61 62 70 72 75 80 86 87 90"
            )
        ) else if !CUDA_MAJOR! EQU 11 (
            if !CUDA_MINOR! GEQ 8 (
                set "ARCH_LIST=35 37 50 52 53 60 61 62 70 72 75 80 86 87 90"
            ) else if !CUDA_MINOR! GEQ 5 (
                set "ARCH_LIST=35 37 50 52 53 60 61 62 70 72 75 80 86 87"
            ) else if !CUDA_MINOR! GEQ 1 (
                set "ARCH_LIST=35 37 50 52 53 60 61 62 70 72 75 80 86"
            ) else (
                set "ARCH_LIST=35 37 50 52 53 60 61 62 70 72 75 80"
            )
        ) else if !CUDA_MAJOR! EQU 10 (
            if !CUDA_MINOR! GEQ 2 (
                set "ARCH_LIST=30 35 37 50 52 53 60 61 62 70 72 75"
            ) else (
                set "ARCH_LIST=30 35 37 50 52 53 60 61 62 70 72 75"
            )
        ) else (
            echo WARNING: CUDA 10.2 or greater is recommended, but attempting to build anyways.
            set "ARCH_LIST=30 37 53 70"
        )
        
        REM --- Step 3: Generate -gencode flags from ARCH_LIST ---
        for %%a in (!ARCH_LIST!) do (
            set "CUDA_ARCH_FLAGS=!CUDA_ARCH_FLAGS! -gencode arch=compute_%%a,code=sm_%%a"
        )
        echo Using CUDA Architectures:!CUDA_ARCH_FLAGS!

    ) else (
        echo WARNING: nvcc not found or version could not be determined.
        echo Will attempt to compile without specific CUDA architecture flags.
    )

    REM NVCC-specific options including the new architecture flags
    set "COMPILER_SPECIFIC_OPTIONS=-DUSE_CUDA -DENABLE_FP16 !CUDA_ARCH_FLAGS! -lcudart -lcublas -Xcompiler "/O2 /std:c++14""

    REM Collect all .cc and .cu files for nvcc
    for %%f in (
        "%SRC_DIR%\*.cc"
        "%BENCHMARK_DIR%\*.cc"
        "%GAME_DIR%\*.cc"
        "%NEURAL_DIR%\*.cc"
        "%NEURAL_BLAS_DIR%\*.cc"
        "%NEURAL_CUDA_DIR%\*.cc"
        "%NEURAL_CUDA_DIR%\*.cu"
        "%PATTERN_DIR%\*.cc"
        "%MCTS_DIR%\*.cc"
        "%SELFPLAY_DIR%\*.cc"
        "%UTILS_DIR%\*.cc"
    ) do (
        for %%g in (%%f) do (
            set "FILES=!FILES! "%%~g""
        )
    )
) else if /I "%COMPILER_CHOICE%"=="gcc" (
    set "COMPILER_CMD=g++"
    REM GCC-specific options
    set "COMPILER_SPECIFIC_OPTIONS=-ffast-math -lpthread -I third_party\Eigen -DUSE_BLAS -DUSE_EIGEN -std=c++14 -static"

    REM Collect all .cc files for gcc (excluding NEURAL_CUDA_DIR)
    for %%f in (
        "%SRC_DIR%\*.cc"
        "%BENCHMARK_DIR%\*.cc"
        "%GAME_DIR%\*.cc"
        "%NEURAL_DIR%\*.cc"
        "%NEURAL_BLAS_DIR%\*.cc"
        "%PATTERN_DIR%\*.cc"
        "%MCTS_DIR%\*.cc"
        "%SELFPLAY_DIR%\*.cc"
        "%UTILS_DIR%\*.cc"
    ) do (
        for %%g in (%%f) do (
            set "FILES=!FILES! "%%~g""
        )
    )
) else (
    echo Invalid COMPILER_CHOICE: %COMPILER_CHOICE%. Please set it to 'gcc' or 'nvcc'.
    goto :eof
)

REM Combine common and compiler-specific options
set "FULL_OPTIONS=%COMMON_OPTIONS% %COMPILER_SPECIFIC_OPTIONS%"

REM Call the selected compiler
echo.
echo Compiling with %COMPILER_CMD%...
%COMPILER_CMD% %FILES% -o %OUTPUT% %FULL_OPTIONS%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Compile successful: %OUTPUT%
) else (
    echo.
    echo Compile failed.
)

endlocal
pause