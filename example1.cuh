#ifndef _EXAMPLE1_CUH
#define _EXAMPLE1_CUH

#include "mathstack.cuh"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

template<class I, class F>
__global__ 
void example1_kernel(I* stack, I stacksize, I* opstack, I opstacksize,
    F* valuestack, I valuestacksize, F* outputstack, I outputstacksize, I Nthreads) 
{
    I s_size = stacksize;
    I op_size = opstacksize;
    I v_size = valuestacksize;
    I ou_size = outputstacksize;

    unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;

    Vars<F> Variables;
    Variables.a = 1.569492;
    Variables.b = 1.5;

    for(int i=0;i<1000;i++) {
        F test = evaluateStackExpr(stack,s_size,opstack,op_size,
            valuestack, v_size, outputstack, ou_size, tid, Nthreads, Variables);
    }

}

#endif