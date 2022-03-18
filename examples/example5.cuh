#ifndef _EXAMPLE1_CUH
#define _EXAMPLE1_CUH

#include "mathstack.cuh"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

template<class F>
__global__ 
void example5_kernel(int* stack, int stacksize, long long* opstack, long long opstacksize,
    F* valuestack, int valuestacksize, double* outputstack, int outputstacksize, int Nthreads) 
{
    int s_size = stacksize;
    long long op_size = opstacksize;
    int v_size = valuestacksize;
    int ou_size = outputstacksize;

    unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;

    Vars<F> Variables;
    Variables.a = 1.569492;

    for(int i=0;i<1;i++) {
        F test = evaluateStackExpr(stack,s_size,opstack,op_size,
            valuestack, v_size, outputstack, ou_size, tid, Nthreads, Variables);
    }

}

#endif