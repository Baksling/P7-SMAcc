#ifndef SIMULATOR_TOOLS_H
#define SIMULATOR_TOOLS_H

#include "../Domain/common.h"
#include <iostream>
#include <map>
#include "simulation_strategy.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#define QUALIFIERS static __forceinline__ __host__ __device__
#include <curand_kernel.h>
#undef QUALIFIERS


class bit_handler
{
public:
    CPU GPU static bool bit_is_set(const unsigned long long* n, unsigned int i);
    CPU GPU static void set_bit(unsigned long long* n, unsigned int i);
    CPU GPU static void unset_bit(unsigned long long* n, unsigned int i);  
};


class result_handler
{
public:
    static void read_results(const int* cuda_results, unsigned long total_simulations, std::map<int, unsigned long>* results);
    static float calc_percentage(unsigned long counter, unsigned long divisor);
    static void print_results(std::map<int,unsigned long>* result_map, unsigned long result_size);
};


#endif