#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define N 8
#define BLOCK_SIZE 2

__global__ void matrixSum(float *d_a, float *d_b, float *d_c){
    int globalIndex = blockIdx.y * BLOCK_SIZE * N + blockIdx.x * BLOCK_SIZE + threadIdx.y * N + threadIdx.x;

    d_c[globalIndex] = d_a[globalIndex] + d_b[globalIndex];
}

int main(){
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    int memSize = N * N * sizeof(float);

    h_a = (float*) malloc(memSize);
    h_b = (float*) malloc(memSize);
    h_c = (float*) malloc(memSize);

    cudaMalloc((void**) &d_a, memSize);
    cudaMalloc((void**) &d_b, memSize);
    cudaMalloc((void**) &d_c, memSize);

    for (int i = 0; i < N * N; ++i) {
        h_a[i] = h_b[i] = 1.0f;
    }

    cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, memSize, cudaMemcpyHostToDevice);

    dim3 block(N / BLOCK_SIZE, N / BLOCK_SIZE);
    dim3 thread(BLOCK_SIZE, BLOCK_SIZE);

    matrixSum<<< block, thread >>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, memSize, cudaMemcpyDeviceToHost);
    printf("El resultado es: \n");

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", h_c[i]);
        }
        printf("\n");
    }



    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}