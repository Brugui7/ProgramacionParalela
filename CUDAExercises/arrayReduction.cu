#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define N 1024


__global__ void arrayReduction(float *d_array, float *d_result){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) d_result[0] += d_array[tid];
}
int main() {
    float *h_array, *h_result;
    float *d_array, *d_result;
    int memSize = sizeof(float) * N;

    h_array = (float*) malloc(memSize);
    h_result = (float*) malloc(sizeof(float));

    cudaError_t error;
    error = cudaMalloc((void**)&d_array, memSize);
    if (error != cudaSuccess){
        fprintf(stderr, "Error al reservar memoria");
        return -1;
    }

    error = cudaMalloc((void**)&d_result, sizeof(float));
    if (error != cudaSuccess){
        fprintf(stderr, "Error al reservar memoria");
        return -1;
    }

    //Fills the arrays
    for (int i = 0; i < N; ++i) {
        h_array[i] = 1.0f;
    }

    //Transfers
    error = cudaMemcpy(d_array, h_array, memSize, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        fprintf(stderr, "Error al transferir información.");
    }
    error = cudaMemcpy(d_result, h_result, sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        fprintf(stderr, "Error al transferir información.");
    }

    dim3 block (N/256);
    dim3 thread (256);

    arrayReduction<<<block, thread>>>(d_array, d_result);

    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("El resultado es: %f\n", h_result[0]);

    cudaFree(d_array);

}