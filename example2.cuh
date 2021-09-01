#ifndef _EXAMPLE1_CUH
#define _EXAMPLE1_CUH

#include "mathstack.cuh"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>


template<class I, class F, class LI>
__global__ 
void example2_kernel(I* stack, I stacksize, LI* opstack, LI opstacksize,
    F* valuestack, I valuestacksize, F* outputstack, I outputstacksize, I Nthreads) 
{
    I s_size    = stacksize;
    LI op_size  = opstacksize;
    I v_size    = valuestacksize;
    I ou_size   = outputstacksize;

    unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;

    Vars<F> Variables;
    Variables.a = 1.569492;
    Variables.b = 1.5;

    F (*sin_ptr)(F) = &sin;
    F (*atan2_ptr)(F,F) = &atan2;
    //opstack[0] = (LI)sin_ptr;
    if(tid==0) {
        printf("sin function addr:   %ld\n",sin_ptr);
        printf("atan2 function addr: %ld\n",atan2_ptr);
    }
    for(int i=0;i<1;i++) {
        F test = evaluateStackExpr(stack,s_size,opstack,op_size,
            valuestack, v_size, outputstack, ou_size, tid, Nthreads, Variables);
    }

}

#endif