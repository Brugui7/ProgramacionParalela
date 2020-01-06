#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void stencil(float *d_a, float *d_b) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0 && tid < N - 1) {
        d_b[tid] = 0.3333f * d_a[tid - 1] * d_a[tid] * d_a[tid + 1];
    }
}

int main() {
    float *h_a, *h_b;
    float *d_a, *d_b;
    int memSize = sizeof(float) * N;

    //Reserve host memory
    h_a = (float *) malloc(memSize);
    h_b = (float *) malloc(memSize);

    //Reserves device memory
    cudaError_t error;
    error = cudaMalloc((void **) &d_a, memSize);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU\n");
        return -1;
    }
    error = cudaMalloc((void **) &d_b, memSize);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error al reservar memoria en la GPU\n");
        return -1;
    }


    //Fills the arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = h_b[i] = 70.0f;
    }

    h_a[0] = h_a[N - 1] = h_b[0] = h_b[N - 1] = 150.0f;

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
    dim3 block(N / 256);
    dim3 thread(256);

    float *aux = NULL;
    for (int i = 0; i < N; ++i) {
        stencil<<<block ,thread>>>(d_a, d_b);
        aux = d_a;
        d_a = d_b;
        d_b = aux;
    }


    error = cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error al transferir información\n");
        return -1;
    }

    for (int i = 0; i < N; ++i) {
        printf("%f, ", h_a[i]);
    }
    printf("\n");

    free(h_a);
    free(h_b);

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}