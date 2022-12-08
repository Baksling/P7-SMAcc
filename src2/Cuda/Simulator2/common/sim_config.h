
#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

#define SHARED_MEMORY_PR_THREAD 32

#include "macro.h"

struct sim_config
{
    //simulation setup
    unsigned int blocks = 1;
    unsigned int threads = 1;
    unsigned int cpu_threads = 1;
    unsigned int simulation_amount = 1;
    unsigned int simulation_repetitions = 1;
    unsigned long long seed = 1;
    int write_mode = 0;
    bool use_max_steps = true;
    unsigned int max_steps_pr_sim = 1;
    double max_global_progression = 1;
    bool verbose = false;
    
    enum device_opt
    {
        device,
        host,
        both
    } sim_location = device;
    
    //model parameters (setup using function)
    bool use_shared_memory = false;
    bool use_jit = false;
    unsigned int max_expression_depth = 1;
    unsigned tracked_variable_count = 1;
    unsigned variable_count = 1;
    unsigned network_size = 1;

    //paths
    std::string model_path;
    std::string output;
    
    //pointers
    void* cache = nullptr;
    curandState* random_state_arr = nullptr;
    
    size_t total_simulations() const
    {
        return static_cast<size_t>(blocks) * threads * simulation_amount;
    }

    bool can_use_cuda_shared_memory(const size_t model_size) const
    {
        return (static_cast<size_t>(this->threads) * SHARED_MEMORY_PR_THREAD) > (model_size);
    }
};

#endif