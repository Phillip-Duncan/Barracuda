/**
 * @file rule110.cu
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Rule110 implementation, for proving MathStack language is turing complete.
 * @version 1.0
 * @date 2021-09-05
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "rule110.cuh"

int main() 
{
    int threads = 1;
    int blocks = 1;
    dim3 Grid(blocks,1,1);
    dim3 Block(1,threads,1);

    float board_size = 50;
    float bs = board_size;
    // 16
    float values[22] = {1,110,1*4,7,1,1*4,
                        (bs-1),1,1*4,1*4,1,10,32,42,1*4,(bs),0,(bs-2),0,1,(bs-2)*4,(bs)*4};

    long ops[38] = {DROP,DROP,SWAP,WRITE,AND,RSHIFT,SWAP,OVER,SUB_P,OVER,OR,READ,OVER,AND,LSHIFT,SWAP,ADD_P,
                        ADD_P,OVER,OR,READ,ADD_P,OVER,LSHIFT,READ,DUP,PRINTC,DROP,PRINTC,TERNARY,READ,DUP,
                        ADD_P,DUP,WRITE,ADD_P,OVER,MALLOC};
    // 43
    int stack[66] = {100,0,0,100,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,1,99,
                        1,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,100,
                        0,0,1,1,0,0,0,1,99,1,1,0,99,1,1,0,1,0,0,1,0,1};
    double output[100*threads*blocks] =   {0};

    // Allocate some memory for stack expressions
    int* stack_dev = NULL;
    int stacksize = 66;
    long* opstack_dev = NULL;
    long opstacksize = 38;
    float* valuesstack_dev = NULL;
    int valuestacksize = 22;
    double* outputstack_dev = NULL;
    int outputstacksize = 0;

    // Increase max stack frame size to max (this has to be increased with increasing threads/blocks).
    cudaError_t stat;
    size_t new_stack_size = 1024*2;
    stat = cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size);
    //size_t max_stack_size[1];
    //cudaDeviceGetLimit(max_stack_size,cudaLimitStackSize);
    //std::cout << max_stack_size[0] << std::endl;

    cudaMalloc((void**)&stack_dev,stacksize*sizeof(int));
    cudaMemcpy(stack_dev,stack,stacksize*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&opstack_dev,opstacksize*sizeof(long));
    cudaMemcpy(opstack_dev,ops,opstacksize*sizeof(long),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&valuesstack_dev,valuestacksize*sizeof(float));
    cudaMemcpy(valuesstack_dev,values,valuestacksize*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&outputstack_dev,100*threads*blocks*sizeof(double));
    cudaMemset(outputstack_dev,0,100*threads*blocks*sizeof(double));


    // Launch example kernel
    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    for (int j=0;j<1;j++) {
        example5_kernel<<<Grid,Block>>>(stack_dev,stacksize,opstack_dev,opstacksize,
        valuesstack_dev,valuestacksize,outputstack_dev,outputstacksize,threads*blocks);
        cudaDeviceSynchronize();
    }

    auto t2 = Clock::now();

    cudaMemcpy(output,outputstack_dev,100*threads*blocks*sizeof(double),cudaMemcpyDeviceToHost);

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




