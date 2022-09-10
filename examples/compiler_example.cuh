#ifndef  _EXAMPLE_COMPILER_CUH
#define _EXAMPLE_COMPILER_CUH


#include "barracuda.cuh"
#include <barracuda_compiler.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

template<class F>
__global__ 
void example_compiler_kernel(int* stack, int stacksize, long long* opstack, int opstacksize,
    F* valuestack, int valuestacksize, double* outputstack, int outputstacksize, int Nthreads,
    Vars<F>* vars) {
    	
	unsigned int tid = (blockIdx.x * blockDim.y) + (blockIdx.y * gridDim.x * blockDim.y) + threadIdx.y;

	Vars<F> thread_variables = vars[tid];

	if(tid==0) {
        #if DEBUG_PRINT_STATE
            printf("Inside Kernel:");
            printf("tid: %d\n", tid);
            printf("Threads: %d\n", Nthreads);

            printf("\nInstructions: ");
            for(int i = 0; i < stacksize; i++) { printf("%d, ", stack[i]);}
            printf("\nOperations: ");
            for(int i = 0; i < opstacksize; i++) { printf("%d, ", opstack[i]);}
            printf("\nValues: ");
            for(int i = 0; i < valuestacksize; i++) { printf("%d, ", (int)valuestack[i]);}
            printf("\nStackSize: %d\n", outputstacksize);
        #endif

		F _test = evaluateStackExpr(stack, stacksize, 
									opstack, opstacksize, 
									valuestack, valuestacksize, 
									outputstack, outputstacksize, 
									tid, Nthreads, thread_variables);
	}
}


#endif // _EXAMPLE_COMPILER_CUH
