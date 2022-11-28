
#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

#include "macro.h"

struct sim_config
{
    //parallelism
    unsigned int blocks{};
    unsigned int threads{};
    unsigned int cpu_threads{};
    unsigned int simulation_amount{};

    //sim configurations
    unsigned long seed{};
    bool use_max_steps = true;
    const unsigned int max_steps_pr_sim{};
    const double max_global_progression{};

    //cache lookup
    int tracked_variable_count{};
    int variable_count{};
    int network_size{};
    
    void* cache{};
    curandState* random_state_arr{};
    
    //Allocation variables
    unsigned int max_expression_depth{};
};

#endif