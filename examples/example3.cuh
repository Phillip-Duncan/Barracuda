#ifndef _EXAMPLE3_CUH
#define _EXAMPLE3_CUH

#include "barracuda.cuh"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

#define MSTACK_SPECIALS 1
#define MSTACK_UNSAFE 1


template<class F>
__device__
F sinhcos(F x) {
    return sinh(x)*cos(x);
}

template<class F>
__global__ 
void example3_kernel(int* stack, int stacksize, long long* opstack, int opstacksize,
    F* valuestack, int valuestacksize, double* outputstack, int outputstacksize, int Nthreads) 
{
    int s_size    = stacksize;
    int op_size  = opstacksize;
    int v_size    = valuestacksize;
    int ou_size   = outputstacksize;

    unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;

    Vars<F> Variables;

    F (*sinhcos_ptr)(F) = &sinhcos;
    if(tid==0) 
        printf("sinhcos function addr:   %lld\n",sinhcos_ptr);

    for(int i=0;i<1;i++) {
        F test = evaluateStackExpr(stack,s_size,opstack,op_size,
            valuestack, v_size, outputstack, ou_size, tid, Nthreads, Variables);
    }

}

#endif