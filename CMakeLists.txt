cmake_minimum_required(VERSION 3.9)
project(Sayuri)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(GAME_SOURCES_DIR ${SOURCE_DIR}/game)
set(UTILS_SOURCES_DIR ${SOURCE_DIR}/utils)
set(GTP_SOURCES_DIR ${SOURCE_DIR}/gtp)


set(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(${IncludePath})

set(BASIC_SOURCES
    ${SOURCE_DIR}/main.cc
    )

set(GTP_SOURCES
    ${GTP_SOURCES_DIR}/gtp.cc
    )

set(GAME_SOURCES
    ${GAME_SOURCES_DIR}/board.cc
    ${GAME_SOURCES_DIR}/simple_board.cc
    ${GAME_SOURCES_DIR}/game_state.cc
    ${GAME_SOURCES_DIR}/string.cc
    ${GAME_SOURCES_DIR}/zobrist.cc
    ${GAME_SOURCES_DIR}/symmetry.cc
    )

set(UTILS_SOURCES
    ${UTILS_SOURCES_DIR}/log.cc
    ${UTILS_SOURCES_DIR}/parser.cc
    ${UTILS_SOURCES_DIR}/random.cc
    )

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

if(DEBUG_MODE)
   message(STATUS "Set Debug Mode")
   set(CMAKE_BUILD_TYPE DEBUG)
else()
   set(CMAKE_BUILD_TYPE RELEASE)
endif()

set(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(GCC_COMPILE_FLAGS "-Wall -Wextra -ffast-math -march=native -flto")
set(GCC_DISABLED_WARNING_COMPILE_FLAGS "-Wno-ignored-attributes -Wno-maybe-uninitialized -Wno-mismatched-tags")
set(CMAKE_CXX_FLAGS "${GCC_COMPILE_FLAGS} ${GCC_DISABLED_WARNING_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -DNDEBUG ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "-flto -g")

add_executable(Sayuri
    ${GTP_SOURCES}
    ${BASIC_SOURCES}
    ${GAME_SOURCES}
    ${UTILS_SOURCES}
    )

target_link_libraries(Sayuri  Threads::Threads)

if(DEBUG_MODE)
   set(CMAKE_BUILD_TYPE DEBUG)
else()
   set(CMAKE_BUILD_TYPE RELEASE)
endif()

