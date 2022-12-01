#ifndef MACRO_H
#define MACRO_H


#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <string>

//HACK TO MAKE CPU WORK!
#define QUALIFIERS static __forceinline__ __host__ __device__
#include <curand_kernel.h>
#undef QUALIFIERS
//HACK SLUT

#define GPU __device__ 
#define CPU __host__
#define GLOBAL __global__
#define IS_GPU __CUDACC__


//While loop done to enfore ; after macro call. See: 
//https://stackoverflow.com/a/61363791/17430854
#define CUDA_CHECK(x)             \
do{                          \
if ((x) != cudaSuccess) {    \
    throw std::runtime_error(std::string("cuda error ") + std::to_string(x) + " in file '" + __FILE__ + "' on line "+  std::to_string(__LINE__)); \
}                             \
}while(0)


__host__ __device__ __forceinline__ void cuda_syncthreads_()
{
    #ifdef __CUDACC__
    #define cuda_SYNCTHREADS() __syncthreads()
    #else
    #define cuda_SYNCTHREADS()
    #endif
} 

#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
#endif

#endif