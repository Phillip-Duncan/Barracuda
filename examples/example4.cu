/**
 * @file example4.cu
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Recursive for loop example.
 * @details Demonstrates that whilst recursion for loops possible, these are probably not a good idea to run,
 * especially with large threads (will likely not be able to alloc enough stack-size for the stack frames required).
 * @version 1.0
 * @date 2021-10-06
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "example4.cuh"

int main() 
{
    int threads = 1;
    int blocks = 1;
    dim3 Grid(blocks,1,1);
    dim3 Block(1,threads,1);

    // Two nested recursive for loops within another for loop. Resultant Sum = 205.
    // Equivalant to: int a=5; for(int i=0;i<10;i++){for(int j=0;j<10;j++){a++;} for(int k=0;k<10;k++){a++;}}
    float values[9] = {1,10,0,1,10,0,10,0,5};
    long ops[2] = {0x3CC,0x3CC};
    int stack[17] = {100,100,0,1,99,1,1,100,0,1,99,1,1,99,1,1,1};
    float output[10*threads*blocks] =   {0};

    // Increase max stack frame size (this has to be increased with increasing threads/blocks).
    cudaError_t stat;
    size_t current_stack_size;
    size_t new_stack_size = 1024*8;
    stat = cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size);
    std::cout << stat << std::endl;

    // Allocate some memory for stack expressions
    int* stack_dev = NULL;
    int stacksize = 17;
    long* opstack_dev = NULL;
    long opstacksize = 2;
    float* valuesstack_dev = NULL;
    int valuestacksize = 9;
    float* outputstack_dev = NULL;
    int outputstacksize = 0;

    cudaMalloc((void**)&stack_dev,stacksize*sizeof(int));
    cudaMemcpy(stack_dev,stack,stacksize*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&opstack_dev,opstacksize*sizeof(long));
    cudaMemcpy(opstack_dev,ops,opstacksize*sizeof(long),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&valuesstack_dev,valuestacksize*sizeof(float));
    cudaMemcpy(valuesstack_dev,values,valuestacksize*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&outputstack_dev,6*threads*blocks*sizeof(float));
    cudaMemset(outputstack_dev,0,6*threads*blocks*sizeof(float));


    // Launch example kernel
    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    for (int j=0;j<1;j++) {
        example4_kernel<<<Grid,Block>>>(stack_dev,stacksize,opstack_dev,opstacksize,
        valuesstack_dev,valuestacksize,outputstack_dev,outputstacksize,threads*blocks);
        cudaDeviceSynchronize();
    }

    auto t2 = Clock::now();

    cudaMemcpy(output,outputstack_dev,6*threads*blocks*sizeof(float),cudaMemcpyDeviceToHost);

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




