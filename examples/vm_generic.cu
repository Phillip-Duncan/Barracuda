/**
 * @file generic.cu
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Generic example.
 * @details Example for generic calls to barracuda VM.
 * @version 1.0
 * @date 2023-03-27
 * 
 * @copyright Copyright (c) 2023-Present
 * 
 */

// Define dllexports for shared libraries
#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT extern "C" __declspec(dllexport)
    #define IMPORT extern "C" __declspec(dllimport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT extern "C" __attribute__((visibility("default")))
    #define IMPORT extern "C" 
#else
    //  do nothing and hope for the best?
    #define EXPORT extern "C"
    #define IMPORT extern "C"
    #pragma warning Unknown dynamic link import/export semantics.
#endif

// Solves issue with identifier "IUnknown" is undefined" on windows
// https://forums.developer.nvidia.com/t/identifier-iunknown-is-undefined-error-vista-visual-studio-2005/3806
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

#include "vm_generic.cuh"

template<class f3264>
void solver(int* instructions, long long* ops, double* values, 
    int stack_size, int output_stack_size, double* userspace, long long* userspace_sizes,
    int blocks, int threads, double* result, double* rt_statistics)
{
    dim3 Grid(blocks,1,1);
    dim3 Block(1,threads,1);

    // Allocate some memory for stack expressions
    int* stack_dev = NULL;
    long long* opstack_dev = NULL;
    double* valuesstack_dev = NULL;
    double* outputstack_dev = NULL;

    cudaMalloc((void**)&stack_dev,stack_size*sizeof(int));
    cudaMemcpy(stack_dev,instructions,stack_size*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&opstack_dev,stack_size*sizeof(long long));
    cudaMemcpy(opstack_dev,ops,stack_size*sizeof(long long),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&valuesstack_dev,stack_size*sizeof(double));
    cudaMemcpy(valuesstack_dev,values,stack_size*sizeof(double),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&outputstack_dev,output_stack_size*threads*blocks*sizeof(double));
    cudaMemset(outputstack_dev,0,output_stack_size*threads*blocks*sizeof(double));

    double* userspace_dev = NULL;
    long long* userspace_sizes_dev = NULL;

    long long total_userspace_size = ((userspace_sizes[0]) * Block.y * Grid.x) + userspace_sizes[1];
    long long userspace_size_mutable = (userspace_sizes[0] - NUM_VARS_TOTAL) * Block.y * Grid.x;


    long long userspace_mutable_offset = NUM_VARS_TOTAL * Block.y * Grid.x;
    long long userspace_constant_offset = (userspace_sizes[0]) * Block.y * Grid.x;

    // Allocate MUTABLE and CONSTANT userspace memory on device
    cudaMalloc((void**)&userspace_dev, total_userspace_size*sizeof(double));
    cudaMemset(userspace_dev, 0, total_userspace_size*sizeof(double));
    cudaMemcpy(userspace_dev + userspace_mutable_offset, userspace, userspace_size_mutable*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(userspace_dev + userspace_constant_offset, userspace + userspace_size_mutable, userspace_sizes[1]*sizeof(double), cudaMemcpyHostToDevice);

    // Allocate sizes of userspace on device
    cudaMalloc((void**)&userspace_sizes_dev, 2*sizeof(long long));
    cudaMemcpy(userspace_sizes_dev, userspace_sizes, 2*sizeof(long long), cudaMemcpyHostToDevice);

    // Set up timer to time kernel execution for statistics
    typedef std::chrono::high_resolution_clock Clock;
    auto timer_start = Clock::now();

    // Launch kernel
    generic_kernel<f3264><<<Grid,Block>>>(stack_dev, stack_size, opstack_dev,
    valuesstack_dev, outputstack_dev, userspace_dev, userspace_sizes_dev, threads*blocks);
    cudaDeviceSynchronize();

    auto timer_end = Clock::now();

    // Calculate time taken and store in runtime statistics.
    rt_statistics[0] = (timer_end - timer_start).count() / 1e9;
    

    // Copy memory back to host and free memory, dont need to copy const memory
    cudaMemcpy(result,outputstack_dev,output_stack_size*threads*blocks*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(userspace, userspace_dev + userspace_mutable_offset, userspace_size_mutable*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(stack_dev);
    cudaFree(opstack_dev);
    cudaFree(valuesstack_dev);
    cudaFree(outputstack_dev);
}


EXPORT void solve_32(int* instructions, long long* ops, double* values, 
        int stack_size, int output_stack_size, double* userspace, long long* userspace_sizes, 
        int blocks, int threads, double* result, double* rt_statistics) 
{
    solver<float>(instructions, ops, values, stack_size, output_stack_size,
            userspace, userspace_sizes, blocks, threads, result, rt_statistics);

}

EXPORT void solve_64(int* instructions, long long* ops, double* values, 
    int stack_size, int output_stack_size, double* userspace, long long* userspace_sizes,
    int blocks, int threads, double* result, double* rt_statistics) 
{
    solver<double>(instructions, ops, values, stack_size, output_stack_size,
            userspace, userspace_sizes, blocks, threads, result, rt_statistics);
}




int main () {
    return 0;
}