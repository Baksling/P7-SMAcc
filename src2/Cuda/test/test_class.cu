#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "dog.h"

dog::dog(int id) {
    this->id_ = id;
}

__host__ __device__ int dog::get_id() {
    return this->id_;
}
__host__ __device__ void dog::bork() {
    printf("%d BORKed!", this->get_id());
}

__global__ void print_dog(dog* dog_, int* out_d) {
    *out_d = dog_->get_id();
    dog_->bork();
}

int main() {

    dog *test; // Host variable
    dog *test_d; // Device Variable

    test = new dog(1);
    
    int *out; // Host variable 
    int *out_d; // Device Variable
    
    out = (int*)malloc(sizeof(int));
    cudaMalloc((void**)&test_d, sizeof(dog));
    cudaMalloc((void**)&out_d, sizeof(int));

    cudaMemcpy(test_d, test, sizeof(dog), cudaMemcpyHostToDevice);

    print_dog<<<1,1>>>(test_d, out_d);

    cudaMemcpy(out, out_d, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("%d", *out);
    
    cudaFree(test_d);
    cudaFree(out_d);

    free(out);

    

    return 0;
}