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
void example1_kernel(int* stack, int stacksize, long long* opstack, int opstacksize,
    double* valuestack, int valuestacksize, double* outputstack, int outputstacksize, int Nthreads,
    Vars* vars) 
{
    int s_size = stacksize;
    int op_size = opstacksize;
    int v_size = valuestacksize;
    int ou_size = outputstacksize;

    unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;

    // Set "first" userspace variable (formerly a)
    Vars Variables = vars[tid];

    Variables.userspace[tid] = 1.569492;

    for(int i=0;i<1000;i++) {
        evaluateStackExpr<F>(stack,s_size,opstack,op_size,
            valuestack, v_size, outputstack, ou_size, tid, Nthreads, Variables);
    }

}

#endif