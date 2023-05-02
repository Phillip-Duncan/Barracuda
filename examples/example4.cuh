#ifndef _EXAMPLE1_CUH
#define _EXAMPLE1_CUH

#include "barracuda.cuh"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

__global__ 
void example4_kernel(int* stack, int stacksize, long long* opstack,
    double* valuestack, double* outputstack, int outputstacksize, int Nthreads) 
{
    int s_size = stacksize;
    int ou_size = outputstacksize;

    unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;

    double* userspace = NULL;

    for(int i=0;i<1;i++) {
        evaluateStackExpr<float>(stack,s_size,opstack, valuestack, 
            outputstack, ou_size, tid, Nthreads, userspace);
    }

}

#endif