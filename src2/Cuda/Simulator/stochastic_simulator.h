
#ifndef STOCHASTIC_SIMULATOR_H
#define STOCHASTIC_SIMULATOR_H


#include "simulation_strategy.h"
#include "../Domain/common.h"
#include "thread_pool.h"
#include <map>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand.h>
#define QUALIFIERS static __forceinline__ __host__ __device__
#include <curand_kernel.h>
#undef QUALIFIERS
#include <chrono>
#include "simulator_tools.h"



class stochastic_simulator
{
public:
    static void simulate_gpu(const stochastic_model_t* model, simulation_strategy* strategy);
    static void simulate_cpu(stochastic_model_t* model, simulation_strategy* strategy);
};

#endif