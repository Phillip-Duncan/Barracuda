#ifndef _EXAMPLE1_CUH
#define _EXAMPLE1_CUH

#include "barracuda.cuh"
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

#define NUM_VARS_TOTAL 2

template<class f3264>
__global__ 
void generic_kernel(int* stack, int stacksize, long long* opstack, int opstacksize,
    double* valuestack, int valuestacksize, double* outputstack, int outputstacksize, Vars* variables_dev,
    int Nthreads) 
{
    int s_size = stacksize;
    int op_size = opstacksize;
    int v_size = valuestacksize;
    int ou_size = outputstacksize;

    unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;

    Vars* stack_vars = &variables_dev[tid];

    f3264 a = 1.57079632679;
    f3264 b = 1.61803398875;

    stack_vars->userspace[tid] = a;
    stack_vars->userspace[tid+Nthreads] = b;

    evaluateStackExpr<f3264>(stack,s_size,opstack,op_size,
        valuestack, v_size, outputstack, ou_size, tid, Nthreads, *stack_vars);


}

#endif