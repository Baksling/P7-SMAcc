#ifndef MACRO_H
#define MACRO_H


#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

//HACK TO MAKE CPU WORK!
#define QUALIFIERS static __forceinline__ __host__ __device__
#include <curand_kernel.h>
#undef QUALIFIERS
//HACK SLUT

#define GPU __device__
#define CPU __host__
#define GLOBAL __global__
#define IS_GPU __CUDACC__

// #define MY_SYNC() ##ifdef __CUDACC__ ##define cuda_SYNCTHREADS() __syncthreads() ##else ##define cuda_SYNCTHREADS() ##endif

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