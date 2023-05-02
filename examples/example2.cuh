#ifndef _EXAMPLE2_CUH
#define _EXAMPLE2_CUH

#include "barracuda.cuh"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

template<class F>
__global__ 
void example2_kernel(int* stack, int stacksize, long long* opstack,
    double* valuestack, double* outputstack, int outputstacksize, int Nthreads,
    double* userspace) 
{
    int s_size    = stacksize;
    int ou_size   = outputstacksize;

    unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;

    userspace[tid] = 1.569492;

    F (*sin_ptr)(F) = &sin;
    F (*atan2_ptr)(F,F) = &atan2;
    //opstack[0] = (long long)sin_ptr;
    if(tid==0) {
        printf("sin function addr:   %lld\n",sin_ptr);
        printf("atan2 function addr: %lld\n",atan2_ptr);
    }
    for(int i=0;i<1;i++) {
        evaluateStackExpr<F>(stack,s_size,opstack, valuestack, 
            outputstack, ou_size, tid, Nthreads, userspace);
    }

}

#endif