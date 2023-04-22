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
    const int threads = 1;
    const int blocks = 1;
    dim3 Grid(blocks,1,1);
    dim3 Block(1,threads,1);

    float board_size = 50;
    float bs = board_size;

    double values[66] = {0,0,0,0,0,0,0,1,0,0,110,0,0,1*4,0,0,0,0,0,7,0,1,0,0,1*4,0,(bs-1),
                        1,0,1*4,0,0,0,0,1*4,0,0,1,0,0,0,10,0,0,0,0,32,42,0,0,0,1*4,0,(bs),
                        0,0,0,(bs-2),0,0,1,0,(bs-2)*4,0,0,(bs+1)*4};

    long long ops[66] = {0,DROP,DROP,0,SWAP,WRITE,AND,0,RSHIFT,SWAP,0,OVER,SUB_P,0,OVER,OR,READ,OVER,AND,0,LSHIFT,0,SWAP,ADD_P,
                        0,0,0,0,ADD_P,0,OVER,OR,READ,ADD_P,0,OVER,LSHIFT,0,READ,DUP,PRINTC,0,DROP,0,PRINTC,TERNARY,0,0,READ,DUP,
                        ADD_P,0,0,0,0,DUP,0,0,0,WRITE,0,ADD_P,0,DUP,MALLOC,0};

    int stack[66] = {100,0,0,100,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,1,99,
                        1,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,100,
                        0,0,1,1,0,0,0,1,99,1,1,0,99,1,1,0,1,0,1,0,0,1};

    double output[100*threads*blocks] =   {0};

    // Allocate some memory for stack expressions
    int* stack_dev = NULL;
    int stacksize = 66;
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

    cudaMalloc((void**)&outputstack_dev,100*threads*blocks*sizeof(double));
    cudaMemset(outputstack_dev,0,100*threads*blocks*sizeof(double));


    // Launch rule110 kernel
    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    for (int j=0;j<1;j++) {
        rule110_kernel<<<Grid,Block>>>(stack_dev,stacksize,opstack_dev,
        valuesstack_dev,outputstack_dev,outputstacksize,threads*blocks);
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




