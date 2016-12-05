cmake_minimum_required(VERSION 3.6)
project(render_gpu)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp kernel_gen.h)

add_custom_command(
        OUTPUT kernel_gen.h
        COMMAND ${CMAKE_C_COMPILER} -E -D__OPENCL_VERSION__ ../render.cl | xxd -i > ../kernel_gen.h
        MAIN_DEPENDENCY ../render.cl)

add_executable(render_gpu ${SOURCE_FILES})

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR})
target_link_libraries(render_gpu ${OPENCL_LIBRARIES} "-framework OpenCL")
