/**
 * @file example2.cu
 * @author Phillip Duncan (phillip.duncan-gelder@pg.canterbury.ac.nz)
 * @brief Function pointers example
 * @version 0.1
 * @date 2021-09-05
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "example2.cuh"

int main() 
{
    const int threads = 1;
    const int blocks = 1;
    dim3 Grid(blocks,1,1);
    dim3 Block(1,threads,1);


    float values[10] = {0,0,0,0,0,0,0,5,6,10};
    long long ops[10] = {471736,MUL,LDNX0,MUL,LDNX0,SIN,ADD,0,0,0};
    int stack[10] = {-2,0,0,0,0,0,0,1,1,1};
    double output[6*threads*blocks] =   {0};

    // Allocate some memory for stack expressions
    int* stack_dev = NULL;
    int stacksize = 10;
    long long* opstack_dev = NULL;
    int opstacksize = 10;
    float* valuesstack_dev = NULL;
    int valuestacksize = 10;
    double* outputstack_dev = NULL;
    int outputstacksize = 0;

    cudaMalloc((void**)&stack_dev,stacksize*sizeof(int));
    cudaMemcpy(stack_dev,stack,stacksize*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&opstack_dev,opstacksize*sizeof(long long));
    cudaMemcpy(opstack_dev,ops,opstacksize*sizeof(long long),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&valuesstack_dev,valuestacksize*sizeof(float));
    cudaMemcpy(valuesstack_dev,values,valuestacksize*sizeof(float),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&outputstack_dev,6*threads*blocks*sizeof(double));
    cudaMemset(outputstack_dev,0,6*threads*blocks*sizeof(double));


    // Allocate some user-space
    int user_space_size = 64*threads*blocks;

    float* user_space_dev = NULL; 
    cudaMalloc((void**)&user_space_dev,user_space_size*sizeof(float));
    cudaMemset((void**)&user_space_dev,0,user_space_size*sizeof(float));

    Vars<float> vars;
    Vars<float>* variables_host = (Vars<float>*)malloc(threads* blocks * sizeof(vars));
    Vars<float>* variables_dev = NULL;
    

    for (int i=0; i<threads*blocks; i++) {
        variables_host[i].userspace = user_space_dev;
    }

    cudaMalloc((void**)&variables_dev,threads*blocks*sizeof(vars));
    cudaMemcpy(variables_dev, variables_host, threads*blocks*sizeof(vars), cudaMemcpyHostToDevice);


    // Launch example kernel
    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    for (int j=0;j<1;j++) {
        example2_kernel<<<Grid,Block>>>(stack_dev,stacksize,opstack_dev,opstacksize,
        valuesstack_dev,valuestacksize,outputstack_dev,outputstacksize,threads*blocks, variables_dev);
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




