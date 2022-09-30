﻿#pragma once
#include "stochastic_model_t.h"

struct model_options
{
    unsigned int simulation_amount;
    unsigned int max_steps_pr_sim;
    unsigned long seed;
};

struct simulation_strategy
{
    int parallel_degree;
    int threads_n;
    unsigned int simulation_amounts;
    int sim_count;
    unsigned int max_sim_steps;

    int total_simulations() const
    {
        return parallel_degree*threads_n*static_cast<int>(simulation_amounts);
    }
};

class cuda_simulator
{
public:
    void simulate(const stochastic_model_t* model, simulation_strategy* strategy);
};