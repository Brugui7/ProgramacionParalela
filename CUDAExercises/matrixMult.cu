#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 8
#define BLOCK_SIZE 2

__global__ void matrixMult(float *d_a, float *d_b, float *d_c){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < N && col < N){
        float sumAux = 0.0f;
        for (int i = 0; i < N; ++i) {
            sumAux += d_a[row * N + i] * d_b[i * N + col];
        }
        d_c[row * N + col] = sumAux;
    }

}

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    int memSize = N * N * sizeof(float);

    h_a = (float*) malloc(memSize);
    h_b = (float*) malloc(memSize);
    h_c = (float*) malloc(memSize);

    cudaError_t status;
    status = cudaMalloc((void**)&d_a, memSize);
    if (status != cudaSuccess) {
        printf("Error");
    }
    status = cudaMalloc((void**)&d_b, memSize);
    if (status != cudaSuccess) {
        printf("Error");
    }
    status = cudaMalloc((void**)&d_c, memSize);
    if (status != cudaSuccess) {
        printf("Error");
    }

    for (int i = 0; i < N * N; ++i) {
        h_a[i] = h_b[i] = 2.0f;
    }

    status = cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        printf("Error passing info HtD");
    }
    status = cudaMemcpy(d_b, h_b, memSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        printf("Error passing info HtD");
    }
    status = cudaMemcpy(d_c, h_c, memSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        printf("Error passing info HtD");
    }


    dim3 block(N / BLOCK_SIZE, N / BLOCK_SIZE);
    dim3 thread(BLOCK_SIZE, BLOCK_SIZE);

    matrixMult<<< block, thread >>>(d_a, d_b, d_c);

    status = cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        printf("Error passing info DtH");
    }
    status = cudaMemcpy(h_b, d_b, memSize, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        printf("Error passing info DtH");
    }
    status = cudaMemcpy(h_c, d_c, memSize, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        printf("Error passing info DtH");
    }

    printf("Resultado: \n");
    for (int i = 0; i < N; ++i) {
        printf("\n");
        for (int j = 0; j < N; ++j) {
            printf("%f\t", h_c[i * N + j]);
        }
    }


}