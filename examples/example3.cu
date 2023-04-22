/**
 * @file example3.cu
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Special functions example (integration)
 * @version 0.1
 * @date 2021-09-05
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "example3.cuh"

int main() 
{
    const int threads = 1;
    const int blocks = 1;
    dim3 Grid(blocks,1,1);
    dim3 Block(1,threads,1);

    // intmethod, maxsteps, accuracy, functype, function, upperlim, lowerlim
    double values[8] = {0,1,1000,0.0001,1,387288,3.141,0}; // 0x7E3
    long long ops[8] = {0xF30,0,0,0,0,0,0,0};
    int stack[8] = {0,1,1,1,1,1,1,1};
    double output[10*threads*blocks] =   {0};

    // Allocate some memory for stack expressions
    int* stack_dev = NULL;
    int stacksize = 8;
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

    cudaMalloc((void**)&outputstack_dev,10*threads*blocks*sizeof(double));
    cudaMemset(outputstack_dev,0,10*threads*blocks*sizeof(double));

    // Launch example kernel
    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    for (int j=0;j<1;j++) {
        example3_kernel<<<Grid,Block>>>(stack_dev,stacksize,opstack_dev,
        valuesstack_dev,outputstack_dev,outputstacksize,threads*blocks);
        cudaDeviceSynchronize();
    }

    auto t2 = Clock::now();

    cudaMemcpy(output,outputstack_dev,10*threads*blocks*sizeof(double),cudaMemcpyDeviceToHost);

    std::cout << "outputs: ";
    for (int i=0;i<10;i++) {
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




