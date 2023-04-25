/**
 * @file generic.cu
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Generic example.
 * @details Example for generic calls to barracuda VM.
 * @version 1.0
 * @date 2023-03-27
 * 
 * @copyright Copyright (c) 2023
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
    int stack_size, int output_stack_size, double* user_space, int user_space_size,
    int blocks, int threads, double* result)
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

    double* user_space_dev = NULL;

    int total_user_space_size = (NUM_VARS_TOTAL + user_space_size) * Block.y * Grid.x;
    int user_space_size_threaded = user_space_size * Block.y * Grid.x;

    int user_space_offset = NUM_VARS_TOTAL * Block.y * Grid.x;

    cudaMalloc((void**)&user_space_dev, total_user_space_size*sizeof(double));
    cudaMemset((void**)&user_space_dev, 0, total_user_space_size*sizeof(double));
    cudaMemcpy(user_space_dev + user_space_offset, user_space, user_space_size_threaded*sizeof(double), cudaMemcpyHostToDevice);



    // Launch example kernel
    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();
    generic_kernel<f3264><<<Grid,Block>>>(stack_dev, stack_size, opstack_dev,
    valuesstack_dev, outputstack_dev, output_stack_size, user_space_dev, threads*blocks);
    cudaDeviceSynchronize();
    auto t2 = Clock::now();

    cudaMemcpy(result,outputstack_dev,output_stack_size*threads*blocks*sizeof(double),cudaMemcpyDeviceToHost);


    std::cout << "\n Elapsed time: " << (t2-t1).count()/1e9 << " s" << std::endl;
    // Free memory

    cudaFree(stack_dev);
    cudaFree(opstack_dev);
    cudaFree(valuesstack_dev);
    cudaFree(outputstack_dev);


}


EXPORT void solve_32(int* instructions, long long* ops, double* values, 
        int stack_size, int output_stack_size, double* user_space, int user_space_size, 
        int blocks, int threads, double* result) 
{
    solver<float>(instructions, ops, values, stack_size, output_stack_size,
            user_space, user_space_size, blocks, threads, result);

}

EXPORT void solve_64(int* instructions, long long* ops, double* values, 
    int stack_size, int output_stack_size, double* user_space, int user_space_size,
    int blocks, int threads, double* result) 
{
    solver<double>(instructions, ops, values, stack_size, output_stack_size,
            user_space, user_space_size, blocks, threads, result);
}




int main () {
    return 0;
}