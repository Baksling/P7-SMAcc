#pragma once

#ifndef CUDA_SIM_H
#define CUDA_SIM_H

#include <unordered_map>

#include "common.h"


struct model_options
{
    unsigned int simulation_amount;
    unsigned int max_steps_pr_sim;
    unsigned long seed;
};

struct simulation_strategy
{
    int block_n = 2560;
    int threads_n = 512;
    unsigned int simulation_amounts = 26;
    int sim_count = 1;
    unsigned int max_sim_steps = 1000;

    int degree_of_parallelism() const
    {
        return block_n*threads_n;
    }
    
    unsigned long total_simulations() const
    {
        return block_n*threads_n*simulation_amounts;
    }
};

class cuda_simulator
{
public:
    static void simulate(stochastic_model_t* model, simulation_strategy* strategy);
};
#endif
