#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
using namespace std;

// cpu implementation of vector addition
void vecAdd_host(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// gpu implementation of vector addition
__global__ 
void vecAdd_kernel(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }    
}

int main(int argc, char* argv[]){
    int n = atoi(argv[1]);
    int size = n*sizeof(float);
    float *A_h, *B_h, *C_h;
    A_h = (float*) malloc(size);
    B_h = (float*) malloc(size);
    C_h = (float*) malloc(size);


    for (int i=0;i<n;i++){
        A_h[i] = i;
        B_h[i] = 2*i;

    };

    auto start = std::chrono::high_resolution_clock::now();
    vecAdd_host(A_h, B_h, C_h, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> host_time = end - start;
    std::cout << "Time taken by the host: " << host_time.count() << "seconds\n";
    
    float *A_d, *B_d, *C_d;
    
    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    int blocksize = 256;
    int gridsize = (n+blocksize-1)/blocksize;
    auto start_d = std::chrono::high_resolution_clock::now();
    vecAdd_kernel<<<gridsize, blocksize >>>(A_d, B_d, C_d, n);
    cudaDeviceSynchronize();
    auto end_d = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> device_time = end_d - start_d;
    std::cout << "Time taken by the device: "<< device_time.count() << "seconds\n";
    cudaMemcpy(C_h, C_d, size,  cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}