cmake_minimum_required(VERSION 3.8)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

project(MathStack VERSION 1.0)

enable_language(CUDA)
find_package(CUDA)

add_definitions(-DMSTACK_SPECIALS=0
-DMSTACK_UNSAFE=1
-DMSTACK_LOOP_NEST_LIMIT=24)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMPILE_DEFINITIONS} -Xcompiler -fPIC -O3 -use_fast_math")

add_subdirectory(examples)


