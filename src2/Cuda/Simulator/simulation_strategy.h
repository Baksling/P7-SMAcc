#ifndef SIMULATION_STRATEGY_H
#define SIMULATION_STRATEGY_H

#define HIT_MAX_STEPS (-1)


struct model_options
{
    unsigned int simulation_amount;
    unsigned int max_steps_pr_sim;
    unsigned long seed;
    const double max_global_progression;
    unsigned int max_expression_depth;
};



struct simulation_strategy
{
    int block_n = 2560;
    int threads_n = 512;
    unsigned int simulation_amounts = 26;
    unsigned int cpu_threads_n = 8;
    int sim_count = 1;
    unsigned int max_sim_steps = 1000;
    double max_time_progression = 10.0;

    int degree_of_parallelism() const
    {
        return block_n*threads_n;
    }
    
    unsigned long total_simulations() const
    {
        return block_n*threads_n*simulation_amounts;
    }
};

#endif