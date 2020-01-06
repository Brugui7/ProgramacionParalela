#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define N 1024

__global__ void arraySum (float *d_a, float *d_b, float *d_c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N){
        d_c[tid] = d_a[tid] + d_b[tid];
    }
}

int main(){
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int memSize = sizeof(float) * N;

    //Reserve host memory
    h_a = (float*) malloc(memSize);
    h_b = (float*) malloc(memSize);
    h_c = (float*) malloc(memSize);

    //Reserves device memory
    cudaError_t error;
    error = cudaMalloc((void**)&d_a, memSize);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU\n");
        return -1;
    }
    error = cudaMalloc((void**)&d_b, memSize);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU\n");
        return -1;
    }
    error = cudaMalloc((void**)&d_c, memSize);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU\n");
        return -1;
    }

    //Fills the arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = h_b[i] = 1.0f;
    }

    //Copies host memory to device
    error = cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error al transferir información\n");
        return -1;
    }

    error = cudaMemcpy(d_b, h_b, memSize, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error al transferir información\n");
        return -1;
    }

    //Grid Definition
    dim3 block (N/256);
    dim3 thread (256);

    arraySum<<< block, thread >>>(d_a, d_b, d_c);

    error = cudaMemcpy(h_c, d_c, memSize, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error al transferir información\n");
        return -1;
    }

    for (int i = 0; i < N; ++i) {
        printf("%f, ", h_c[i]);
    }
    printf("\n");
    return 0;
}