cmake_minimum_required(VERSION 3.9)
project(Sayuri)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(GAME_SOURCES_DIR ${SOURCE_DIR}/game)
set(UTILS_SOURCES_DIR ${SOURCE_DIR}/utils)

set(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(${IncludePath})

set(BASIC_SOURCES
    ${SOURCE_DIR}/main.cc
    )

set(GAME_SOURCES
    ${GAME_SOURCES_DIR}/board.cc
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

add_executable(Sayuri
    ${BASIC_SOURCES}
    ${GAME_SOURCES}
    ${UTILS_SOURCES}
    )

target_link_libraries(Sayuri  Threads::Threads)
