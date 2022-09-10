include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(example1 example1.cu example1.cuh)
target_compile_definitions(example1 PRIVATE COMPILE_DEFINITIONS)

add_executable(example2 example2.cu example2.cuh)


add_executable(example3 example3.cu example3.cuh)


add_executable(example4 example4.cu example4.cuh)


add_executable(example5 example5.cu example5.cuh)


add_executable(rule110 rule110.cu rule110.cuh)


# Compiler Example
include_directories(${PROJECT_SOURCE_DIR}/examples/include)
add_executable(compiler_example compiler_example.cu compiler_example.cuh include/barracuda_compiler.h)
target_link_libraries(compiler_example ${PROJECT_SOURCE_DIR}/examples/lib/libbarracuda_compiler.so)
