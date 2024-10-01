include_directories(${PROJECT_SOURCE_DIR}/src)

# Only 1 example now, generic VM implementation, tested externally (makes it simpler to test, debug).
add_library(vm_generic SHARED vm_generic.cu vm_generic.cuh)
