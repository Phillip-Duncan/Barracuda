#ifndef _EXAMPLE1_CUH
#define _EXAMPLE1_CUH

#include "barracuda.cuh"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

template<class F>
__global__
void example1_kernel(int* stack, int stacksize, long long* opstack,
    double* valuestack, double* outputstack, int outputstacksize, int Nthreads,
    double* userspace) 
{
    int s_size = stacksize;
    int ou_size = outputstacksize;

    unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;

    userspace[tid] = 1.5;

    for(int i=0;i<1000;i++) {
        evaluateStackExpr<F>(stack,s_size,opstack, valuestack,
            outputstack, ou_size, tid, Nthreads, userspace);
    }

}

#endif