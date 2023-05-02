#ifndef _EXAMPLE1_CUH
#define _EXAMPLE1_CUH

#include "barracuda.cuh"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

#define NUM_VARS_TOTAL 2

template<class f3264>
__global__ 
void generic_kernel(int* stack, int stacksize, long long* opstack,
    double* valuestack, double* outputstack, int outputstacksize, double* userspace,
    int Nthreads) 
{
    int s_size = stacksize;
    int ou_size = outputstacksize;

    unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;


    f3264 a = 1.57079632679;
    f3264 b = 1.61803398875;

    userspace[tid] = a;
    userspace[tid+Nthreads] = b;

    evaluateStackExpr<f3264>(stack,s_size,opstack, valuestack, 
        outputstack, ou_size, tid, Nthreads, userspace);
}

#endif