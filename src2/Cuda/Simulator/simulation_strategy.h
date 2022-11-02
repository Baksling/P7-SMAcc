#ifndef SIMULATION_STRATEGY_H
#define SIMULATION_STRATEGY_H


#define HIT_MAX_STEPS (-1)
#include "../Domain/clock_variable.h"
class node_t;

struct model_state
{
    node_t* current_node;
    bool reached_goal;
};

struct model_options
{
    unsigned int simulation_amount{};
    unsigned long seed{};
    bool use_max_steps = true;
    const unsigned int max_steps_pr_sim{};
    const double max_global_progression{};


    //Allocation variables
    unsigned int max_expression_depth{};
    unsigned int model_count{};
    unsigned int variable_count{};
    unsigned int timer_count{};

    CPU GPU unsigned int get_expression_size() const
    {
        return this->max_expression_depth*2+1;
    }
    
    CPU GPU unsigned long long int get_cache_size() const
    {
        const unsigned long long size =
              this->get_expression_size() * sizeof(void*) + //this is a expression*, but it doesnt like sizeof(expression*)
              max_expression_depth * sizeof(double) +
              model_count * sizeof(model_state) +
              variable_count * sizeof(clock_variable) +
              timer_count * sizeof(clock_variable);

        const unsigned long long int padding = (8 - (size % 8));

        return padding < 8 ? size + padding : size;
    }
};



struct simulation_strategy
{
    unsigned int block_n = 1;
    unsigned int threads_n = 1;
    unsigned int simulations_per_thread = 1;
    unsigned int cpu_threads_n = 1;
    unsigned int simulation_runs = 1;

    bool use_max_steps = true;
    unsigned int max_sim_steps = 100;
    double max_time_progression = 100.0;

    unsigned int degree_of_parallelism() const
    {
        return block_n*threads_n;
    }
    
    unsigned long total_simulations() const
    {
        return block_n*threads_n*simulations_per_thread;
    }
};

#endif