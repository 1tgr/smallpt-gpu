cmake_minimum_required(VERSION 3.6)
project(render_gpu)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(render_gpu ${SOURCE_FILES})