/**
 * @file example1.cu
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Simple arithmetic example
 * @version 0.1
 * @date 2021-09-05
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "example1.cuh"

int main() 
{
    int threads = 256;
    int blocks = 256;
    dim3 Grid(blocks,1,1);
    dim3 Block(1,threads,1);


    float values[3] = {5,6,10};
    int ops[7] = {0x3CF,0x3CE,0x12FD,0x3CE,0x12FD,0x7E4,0x3CC};
    int stack[10] = {0,0,0,0,0,0,0,1,1,1};
    float output[6*threads*blocks] =   {0};

    // Allocate some memory for stack expressions
    int* stack_dev = NULL;
    int stacksize = 10;
    int* opstack_dev = NULL;
    int opstacksize = 7;
    float* valuesstack_dev = NULL;
    int valuestacksize = 3;
    float* outputstack_dev = NULL;
    int outputstacksize = 0;

    cudaMalloc((void**)&stack_dev,stacksize*sizeof(int));
    cudaMemcpy(stack_dev,stack,stacksize*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&opstack_dev,opstacksize*sizeof(int));
    cudaMemcpy(opstack_dev,ops,opstacksize*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&valuesstack_dev,valuestacksize*sizeof(float));
    cudaMemcpy(valuesstack_dev,values,valuestacksize*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&outputstack_dev,6*threads*blocks*sizeof(float));
    cudaMemset(outputstack_dev,0,6*threads*blocks*sizeof(float));


    // Launch example kernel
    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    for (int j=0;j<50;j++) {
        example1_kernel<<<Grid,Block>>>(stack_dev,stacksize,opstack_dev,opstacksize,
        valuesstack_dev,valuestacksize,outputstack_dev,outputstacksize,threads*blocks);
        cudaDeviceSynchronize();
    }

    auto t2 = Clock::now();

    cudaMemcpy(output,outputstack_dev,6*threads*blocks*sizeof(float),cudaMemcpyDeviceToHost);

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




