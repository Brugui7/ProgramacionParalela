#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define N 1024

__global__ void saxpy(float *d_x, float *d_y){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) d_y[tid] = d_x[tid] * 2.0f + d_y[tid];
}

int main(){
    float *h_y, *h_x;
    float *d_y, *d_x;
    int memSize = sizeof(float) * N;
    h_y = (float*) malloc(memSize);
    h_x = (float*) malloc(memSize);
    cudaMalloc((void**)&d_x, memSize);
    cudaMalloc((void**)&d_y, memSize);

    for (int i = 0; i < N; ++i) {
        h_x[i] = h_y[i] = 1.0f;
    }

    cudaMemcpy(d_x, h_x, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, memSize, cudaMemcpyHostToDevice);

    dim3 block(N / 256);
    dim3 thread(256);
    saxpy<<< block, thread >>>(d_x, d_y);

    cudaMemcpy(h_x, d_x, memSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, memSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        printf("%f\n", h_y[i]);
    }

    free(h_y);
    free(h_x);
    cudaFree(d_x);
    cudaFree(d_y);


}