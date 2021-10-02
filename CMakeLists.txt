cmake_minimum_required(VERSION 3.8)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

project(MathStack VERSION 1.0)

enable_language(CUDA)
find_package(CUDA)


add_subdirectory(examples)


