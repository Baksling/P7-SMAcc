#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define N 10000000

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    //cudaSetDevice(1);
    // int deviceCount;
    // cudaGetDeviceCount(&deviceCount);
    // int device;
    // for (device = 0; device < deviceCount; ++device) {
    //     cudaDeviceProp deviceProp;
    //     cudaGetDeviceProperties(&deviceProp, device);
    //     printf("Device %d has compute capability %d.%d.\n",
    //     device, deviceProp.major, deviceProp.minor);
    // }
    float *a, *b, *out; 
    float *a_d, *b_d, *out_d;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }
    
    cudaMalloc((void**)&a_d, sizeof(float) * N);
    cudaMalloc((void**)&b_d, sizeof(float) * N);
    cudaMalloc((void**)&out_d, sizeof(float) * N);

    
    cudaMemcpy(a_d, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Main function
    vector_add<<<1,1024>>>(out_d, a_d, b_d, N);

    cudaMemcpy(out, out_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%f \n", out[i]);
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(out_d);

    free(a);
    free(b);
    free(out);
}