/**
 * @file example5.cu
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Jumping examples.
 * @details Use of conditional and non-conditional jumps.
 * @version 1.0
 * @date 2021-10-06
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "example5.cuh"

int main() 
{
    const int threads = 128;
    const int blocks = 128;
    dim3 Grid(blocks,1,1);
    dim3 Block(1,threads,1);


    long long jumppos1l = 7;
    long long jumppos2l = 3;
    long long jumppos3l = 10;

    double jp1 = *(double*)(void*)&jumppos1l;
    double jp2 = *(double*)(void*)&jumppos2l;
    double jp3 = *(double*)(void*)&jumppos3l;

    // Two unconditional jumps and a single conditional jump.
    // First jump brings sp down to conditonal jump, where conditionally jumps up to operation +1 and then
    // jumps to end of program.
    //float values[6] = {10,1,3,0,7,5};
    //long long ops[1] = {0x3CC};
    double values[11] = {0,0,jp2,0,0,jp3,0,1,0,jp1,5};
    long long ops[11] = {0,0,0,0,0,0,0x3CC,0,0,0,0};
    int stack[11] = {20,3,1,1,2,1,0,1,2,1,1};
    double output[6*threads*blocks] =   {0};

    // Allocate some memory for stack expressions
    int* stack_dev = NULL;
    int stacksize = 11;
    long long* opstack_dev = NULL;
    int opstacksize = 11;
    double* valuesstack_dev = NULL;
    int valuestacksize = 11;
    double* outputstack_dev = NULL;
    int outputstacksize = 0;

    cudaMalloc((void**)&stack_dev,stacksize*sizeof(int));
    cudaMemcpy(stack_dev,stack,stacksize*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&opstack_dev,opstacksize*sizeof(long long));
    cudaMemcpy(opstack_dev,ops,opstacksize*sizeof(long long),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&valuesstack_dev,valuestacksize*sizeof(double));
    cudaMemcpy(valuesstack_dev,values,valuestacksize*sizeof(double),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&outputstack_dev,6*threads*blocks*sizeof(double));
    cudaMemset(outputstack_dev,0,6*threads*blocks*sizeof(double));


    // Launch example kernel
    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    for (int j=0;j<1;j++) {
        example5_kernel<<<Grid,Block>>>(stack_dev,stacksize,opstack_dev,opstacksize,
        valuesstack_dev,valuestacksize,outputstack_dev,outputstacksize,threads*blocks);
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




