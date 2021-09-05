#ifndef _EXAMPLE3_CUH
#define _EXAMPLE3_CUH

#include "mathstack.cuh"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

template<class F>
__device__
F sinhcos(F x) {
    return sinh(x)*cos(x);
}

template<class I, class F, class LI>
__global__ 
void example3_kernel(I* stack, I stacksize, LI* opstack, LI opstacksize,
    F* valuestack, I valuestacksize, F* outputstack, I outputstacksize, I Nthreads) 
{
    I s_size    = stacksize;
    LI op_size  = opstacksize;
    I v_size    = valuestacksize;
    I ou_size   = outputstacksize;

    unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;

    Vars<F> Variables;

    F (*sinhcos_ptr)(F) = &sinhcos;
    if(tid==0) 
        printf("sinhcos function addr:   %ld\n",sinhcos_ptr);

    for(int i=0;i<1;i++) {
        F test = evaluateStackExpr(stack,s_size,opstack,op_size,
            valuestack, v_size, outputstack, ou_size, tid, Nthreads, Variables);
    }

}

#endif