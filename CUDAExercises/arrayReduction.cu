/**
 * @author Alejandro Brugarolas
 * @since 2019-12
 */
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define N 1024


__global__ void arrayReduction(float *d_array){
    int idx = threadIdx.x;
    int idx2 = 0;

    for (int i = blockDim.x; i >= 1 ; i /=2) {
        if (idx < i){
            idx2 = idx + i;
            d_array[idx] += d_array[idx2];
        }
        __syncthreads();
    }
}
int main() {
    float *h_array;
    float *d_array;
    int memSize = sizeof(float) * N;

    h_array = (float*) malloc(memSize);

    cudaError_t error;
    error = cudaMalloc((void**)&d_array, memSize);
    if (error != cudaSuccess){
        fprintf(stderr, "Error al reservar memoria");
        return -1;
    }


    //Fills the array
    for (int i = 0; i < N; ++i) {
        h_array[i] = 1.0f;
    }

    //Transfers
    error = cudaMemcpy(d_array, h_array, memSize, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        fprintf(stderr, "Error al transferir información.");
    }

    dim3 block (N / (N/2));
    dim3 thread (N/2);

    arrayReduction<<<block, thread>>>(d_array);

    cudaMemcpy(h_array, d_array, sizeof(float), cudaMemcpyDeviceToHost);

    printf("El resultado es: %f\n", h_array[0]);

    cudaFree(d_array);

}