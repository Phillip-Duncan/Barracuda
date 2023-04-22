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

__global__ 
void example3_kernel(int* stack, int stacksize, long long* opstack,
    double* valuestack, double* outputstack, int outputstacksize, int Nthreads) 
{
    int s_size    = stacksize;
    int ou_size   = outputstacksize;

    unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;

    Vars Variables;

    float (*sinhcos_ptr)(float) = &sinhcos;
    if(tid==0) 
        printf("sinhcos function addr:   %lld\n",sinhcos_ptr);

    for(int i=0;i<1;i++) {
        evaluateStackExpr<float>(stack,s_size,opstack, valuestack, 
            outputstack, ou_size, tid, Nthreads, Variables);
    }

}

#endif