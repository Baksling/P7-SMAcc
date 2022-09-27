#pragma once

#ifndef CUDA_SIMULATOR_H
#define CUDA_SIMULATOR_H


#include "uneven_list.h"
#include "stochastic_model.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <unordered_map>

struct simulation_strategy
{
    const int parallel_degree;
    const int threads_n;
    const int simulation_amounts;
    const int sim_count;
    const int max_sim_steps;

    int total_simulations() const
    {
        return parallel_degree*threads_n*simulation_amounts;
    }
};

class cuda_simulator
{

public:
    cuda_simulator();
    void simulate(const stochastic_model* model, const simulation_strategy* strategy) const;
};

#endif // CUDA_SIMULATOR_H