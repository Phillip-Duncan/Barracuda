/**
 * @file example4.cu
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Recursive for loop example.
 * @details Recursive loop example showing power of loop slotting method.
 * @version 1.0
 * @date 2021-10-06
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "example4.cuh"

int main() 
{
    const int threads = 64;
    const int blocks = 64;
    dim3 Grid(blocks,1,1);
    dim3 Block(1,threads,1);

    // Two nested recursive for loops within another for loop. Resultant Sum = 205.
    // Equivalant to: int a=5; for(int i=0;i<10;i++){for(int j=0;j<10;j++){a++;} for(int k=0;k<10;k++){a++;}}

    long long imaxl = 10;
    long long i0l = 0;

    double imax = *(double*)(void*)&imaxl;
    double i0 = *(double*)(void*)&i0l;

    double values[17] = {0,0,0,1,0,imax,i0,0,0,1,0,imax,i0,0,imax,i0,5};
    long long ops[17] = {0,0,0x3CC,0,0,0,0,0,0x3CC,0,0,0,0,0,0,0,0};
    int stack[17] = {100,100,0,1,99,1,1,100,0,1,99,1,1,99,1,1,1};
    double output[10*threads*blocks] =   {0};

    // Allocate some memory for stack expressions
    int* stack_dev = NULL;
    int stacksize = 17;
    long long* opstack_dev = NULL;
    double* valuesstack_dev = NULL;
    double* outputstack_dev = NULL;
    int outputstacksize = 0;

    cudaMalloc((void**)&stack_dev,stacksize*sizeof(int));
    cudaMemcpy(stack_dev,stack,stacksize*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&opstack_dev,stacksize*sizeof(long long));
    cudaMemcpy(opstack_dev,ops,stacksize*sizeof(long long),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&valuesstack_dev,stacksize*sizeof(double));
    cudaMemcpy(valuesstack_dev,values,stacksize*sizeof(double),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&outputstack_dev,6*threads*blocks*sizeof(double));
    cudaMemset(outputstack_dev,0,6*threads*blocks*sizeof(double));


    // Launch example kernel
    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    for (int j=0;j<1;j++) {
        example4_kernel<<<Grid,Block>>>(stack_dev,stacksize,opstack_dev,
        valuesstack_dev,outputstack_dev,outputstacksize,threads*blocks);
        cudaDeviceSynchronize();
    }

    auto t2 = Clock::now();

    cudaMemcpy(output,outputstack_dev,6*threads*blocks*sizeof(double),cudaMemcpyDeviceToHost);

    std::cout << "outputs: ";
    for (int i=0;i<5;i++) {
         std::cout << output[i] << ", ";
    }
    std::cout << std::endl;


    std::cout << "\n Elapsed time: " << (t2-t1).count()/1e9 << " s" << std::endl;
    // Free memory

    cudaFree(stack_dev);
    cudaFree(opstack_dev);
    cudaFree(valuesstack_dev);
    cudaFree(outputstack_dev);

}




