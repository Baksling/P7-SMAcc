#include "cpu_simulator.h"
#include "thread_pool.h"

void cpu_simulator::simulate(stochastic_model_t* model, const simulation_strategy* strategy)
{
    //setup start variables
    const unsigned long total_simulations = strategy->total_simulations();
    //setup random state
    curandState* state = static_cast<curandState*>(malloc(sizeof(curandState) * strategy->degree_of_parallelism()));
    
    //setup results array
    const unsigned long size = sizeof(int)*total_simulations;
    int* sim_results = static_cast<int*>(malloc(size));
    
    printf("allocated %lu (%lu*%lu) bytes successfully: %s\n" ,
        size, static_cast<unsigned long>(sizeof(int)), total_simulations, (sim_results != nullptr ? "True" : "False") );

    //setup simulation options
    const model_options options = {
        strategy->simulation_amounts,
        strategy->max_sim_steps,
        static_cast<unsigned long>(time(nullptr))
    };

    std::map<int, unsigned long> node_results;
    const steady_clock::time_point start = steady_clock::now();
    std::cout << "Started running!\n";
    thread_pool pool;

    pool.start();
    for (int i = 0; i < strategy->degree_of_parallelism(); ++i)
    {
        pool.queue_job([&model, &options, &state, &sim_results, &i]
        {
            simulate_stochastic_model(model, &options, state, sim_results, i);
        });
    }

    while(pool.is_busy())
    {
    }
    pool.stop();

    std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "[ms] \n";
    std::cout << "Reading results...\n";
    read_results(sim_results, total_simulations, &node_results);
    print_results(&node_results, total_simulations * strategy->sim_count);

    free(sim_results);
    free(state);
}
