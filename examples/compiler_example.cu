#include "compiler_example.cuh"

typedef std::chrono::high_resolution_clock Clock;


// Launches cuda kernel with device buffers allocated
// @param response: Successful compiler response loaded with program code
// @param stack_capacity: Each threads stack capacity. Output buffer is Nthreads*stack_capacity.
// @return host buffer of output data from the VM. It is the responsibility of the caller to free this memory
double* launch_kernel(CompilerResponse response, int stack_capacity) {
	// Set thread configuration
	const int threads = 128;
    const int blocks = 128;
    dim3 Grid(blocks,1,1);
    dim3 Block(1,threads,1);


	// Setup Device Buffers
	int* stack_dev = NULL;
	int  stacksize = response.instructions_list.len;
    cudaMalloc((void**)&stack_dev, stacksize*sizeof(int));
    cudaMemcpy(stack_dev, response.instructions_list.ptr, stacksize*sizeof(int), 
    		   cudaMemcpyHostToDevice);

	long long* opstack_dev = NULL;
	int opstacksize = response.operations_list.len;
	cudaMalloc((void**)&opstack_dev, opstacksize*sizeof(long long));
    cudaMemcpy(opstack_dev, response.operations_list.ptr, opstacksize*sizeof(long long),
    		   cudaMemcpyHostToDevice);

	float* valuestack_dev = NULL;
	int valuestacksize = response.values_list.len;
	cudaMalloc((void**)&valuestack_dev, valuestacksize*sizeof(float));
    cudaMemcpy(valuestack_dev, response.values_list.ptr, valuestacksize*sizeof(float),
    		   cudaMemcpyHostToDevice);

	double* outputstack_dev = NULL;
	int outputstacksize = stack_capacity*threads*blocks;
	cudaMalloc((void**)&outputstack_dev, outputstacksize*sizeof(double));
	cudaMemset(outputstack_dev, 0, outputstacksize*sizeof(double));


	// Setup Userspace Buffers
	int user_space_size = 64*threads*blocks;
	float* user_space_dev = NULL;
    cudaMalloc((void**)&user_space_dev,user_space_size*sizeof(float));
    cudaMemset((void**)&user_space_dev,0,user_space_size*sizeof(float));

    Vars<float> vars;
    Vars<float>* variables_host = (Vars<float>*)malloc(threads* blocks * sizeof(vars));
    for (int i=0; i<threads*blocks; i++) {
    	variables_host[i].userspace = user_space_dev;
    }
    
    Vars<float>* variables_dev = NULL;
    cudaMalloc((void**)&variables_dev,threads*blocks*sizeof(vars));
    cudaMemcpy(variables_dev, variables_host, threads*blocks*sizeof(vars), cudaMemcpyHostToDevice);

    // TMP debug

	// Run Kernel
    std::cout << "Starting Kernel" << std::endl;
    auto t1 = Clock::now();

    for (int j=0;j<1;j++) {
        example_compiler_kernel<<<Grid,Block>>>(stack_dev, stacksize, opstack_dev, opstacksize,
        valuestack_dev, valuestacksize, outputstack_dev, 10, threads*blocks, variables_dev);
        cudaDeviceSynchronize();
    }

    auto t2 = Clock::now();
    std::cout << "\nElapsed time: " << (t2-t1).count()/1e9 << " s" << std::endl;

    // Retrieve Output
    double* output = (double*) malloc(outputstacksize*sizeof(double));
    cudaMemcpy(output, outputstack_dev, outputstacksize*sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(stack_dev);
    cudaFree(opstack_dev);
    cudaFree(valuestack_dev);
    cudaFree(outputstack_dev);
    cudaFree(user_space_dev);
    cudaFree(variables_dev);

    return output;
}


// Pretty prints the compiler response data
void print_response(CompilerResponse* response) {
    std::cout << "code_text: " << std::endl;
    std::cout << response->code_text << std::endl;

    std::cout << "instructions: ";
    for (int i=0;i<response->instructions_list.len;i++) {
        std::cout << std::hex << response->instructions_list.ptr[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "operations: ";
    for (int i=0;i<response->operations_list.len;i++) {
        std::cout << std::hex << response->operations_list.ptr[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "values: ";
    for (int i=0;i<response->values_list.len;i++) {
        std::cout << response->values_list.ptr[i] << ", ";
    }
    std::cout << std::endl;
}


int main() {

    // Example barracuda code preloaded as string literal
	const char* program_code =
            "fn fib(n) {"
            "   let a = 0;"
            "   let b = 1;"
            "   for (let i = 0; i < n; i = i + 1) {"
            "       let temp = a + b;"
            "       a = b;"
            "       b = temp;"
            "       print a;"
            "   }"
            "}"
            "fib(10);";


    // Create compile request
	CompilerRequest request {};
	request.code_text = strdup(program_code);
	
	{
		CompilerResponse response = compile(&request);
        print_response(&response);

        double* output = launch_kernel(response, 10);


        // Print non zero outputs in buffer
        std::cout << "outputs: ";
        for (int i = 0 ;i < 10*128*128;i++) {
            if (output[i] != 0) {
                std::cout << i << ": " << output[i] << std::endl;
            }

        }

        free(output);
        free_compile_response(response);
	}
}
