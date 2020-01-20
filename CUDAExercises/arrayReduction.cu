#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define N 1024


__global__ void arrayReduction(float *d_array, float *d_result){
    /*
     * No funciona
    extern __shared__ int sum[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sum[tid] = d_array[i];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0){
            sum[tid] += sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) d_result[blockIdx.x] = sum[0];*/
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) atomicAdd(&d_result[0], d_array[id]);
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