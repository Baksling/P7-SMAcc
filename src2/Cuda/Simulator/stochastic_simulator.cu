#include "stochastic_simulator.h"
#include "thread_pool.h"
#include <map>
#include "device_launch_parameters.h"
#include <chrono>

#include "result_writer.h"
#include "simulator_tools.h"
#include "stochastic_engine.h"
#include <chrono>

#include "../Visitors/domain_analysis_visitor.h"
using namespace std::chrono;

model_options stochastic_simulator::build_options(stochastic_model_t* model,
    const simulation_strategy* strategy)
{
    domain_analysis_visitor analyser = domain_analysis_visitor();
    analyser.visit(model);
    
    return model_options{
        strategy->simulations_per_thread,
        static_cast<unsigned long>(time(nullptr)),

        strategy->use_max_steps,
        strategy->max_sim_steps,
        strategy->max_time_progression,

        analyser.get_max_expression_depth(),
        
        model->get_models_count(),
        model->get_variable_count(),
        model->get_timer_count()
    };
}

void stochastic_simulator::simulate_gpu(stochastic_model_t* model, const simulation_strategy* strategy, result_writer* r_writer, const bool verbose)
{
     //setup start variables
    const unsigned long total_simulations = strategy->total_simulations();

    //setup results array
    const unsigned int variable_count = model->get_variable_count();
    const unsigned int model_count = model->get_models_count();
    
    std::list<void*> free_list;
    std::unordered_map<node_t*, node_t*> node_map;
    const allocation_helper allocator = { &free_list, &node_map };
    
     const simulation_result_container results = simulation_result_container(
        total_simulations,
        model_count,
        variable_count,
        true);
     const simulation_result_container* d_result = results.cuda_allocate(&allocator);
    
    
    //Allocate model on cuda
    stochastic_model_t* model_d = nullptr;
    cudaMalloc(&model_d, sizeof(stochastic_model_t));
    allocator.free_list->push_back(model_d);
    model->cuda_allocate(model_d, &allocator);

    //move options to GPU
    model_options* options_d = nullptr;
    model_options options = build_options(model, strategy);
    cudaMalloc(&options_d, sizeof(model_options));
    cudaMemcpy(options_d, &options, sizeof(model_options), cudaMemcpyHostToDevice);

    //allocate simulation cache.
    const unsigned long long int thread_memory_size = options.get_cache_size();
    void* total_memory_heap;
    cudaMalloc(&total_memory_heap, thread_memory_size * strategy->degree_of_parallelism());
    free_list.push_back(total_memory_heap);
    
    if (verbose) printf("Allocating cache: %llu bytes\n", (thread_memory_size * strategy->degree_of_parallelism()) / 8);

    //run simulations
    if (verbose) std::cout << "Started running!\n";
    const steady_clock::time_point global_start = steady_clock::now();
    
    for (unsigned i = 0; i < strategy->simulation_runs; ++i)
    {
        options.seed = static_cast<unsigned long>(time(nullptr));

        //start time local
        const steady_clock::time_point local_start = steady_clock::now();
        
        //simulate on device
        const bool success = stochastic_engine::run_gpu(model_d, options_d, d_result, strategy, total_memory_heap);
        if(!success)
        {
            printf("An unsuccessful GPU simulation has occured. Stopping simulation.\n");
            break;
        }
    
        //count result unless last sim
        if (verbose)
        {
            std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - local_start).count() << "[ms] \n";
            std::cout << "Reading results...\n";
        }
        
        //simulator_tools::read_results(local_results, total_simulations, model_count, &result_map, &lend_variable_r, true);
        r_writer->write_results(&results, steady_clock::now() - local_start);
    }

    steady_clock::duration temp_time = steady_clock::now() - global_start;
    
    if (verbose) std::cout << "Simulation and result analysis took a total of: " << duration_cast<milliseconds>(temp_time).count() << "[ms] \n";
    r_writer->write_summary(total_simulations,temp_time);
    
    //simulator_tools::print_results(&result_map, &lend_variable_r, total_simulations);
    
    //free local variables
    cudaFree(options_d);
    results.free_internals();
    for (void* it : free_list)
    {
        cudaFree(it);
    }
}

void stochastic_simulator::simulate_cpu(
    stochastic_model_t* model,
    const simulation_strategy* strategy,
    result_writer* r_writer,
    const bool verbose)
{
    //setup start variables
    const unsigned long total_simulations = strategy->total_simulations();

    //setup results array
    const unsigned int variable_count = model->get_variable_count();
    const unsigned int model_count = model->get_models_count();

    //TODO slim this down, this is handled by the result writer
    simulation_result_container results = simulation_result_container(
        total_simulations,
        model_count,
        variable_count,
        false);
    
    model_options options = build_options(model, strategy);

    const unsigned long long int thread_memory_size = options.get_cache_size();
    void* total_memory_heap = malloc(thread_memory_size * strategy->degree_of_parallelism());

    if (verbose) std::cout << "Started running!\n";
    const steady_clock::time_point global_start = steady_clock::now();
    
    for (unsigned i = 0; i < strategy->simulation_runs; ++i)
    {
        options.seed = static_cast<unsigned long>(time(nullptr));
        const steady_clock::time_point local_start = steady_clock::now();
        const bool success = stochastic_engine::run_cpu(model, &options, &results, strategy, total_memory_heap);
        if(!success)
        {
            if (verbose) printf("An unsuccessful CPU simulation has occured. Stopping simulation.");
            break;
        }
        if (verbose)
        {
            std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - local_start).count() << "[ms] \n";
            std::cout << "Reading results...\n";
        }
        r_writer->write_results(&results, steady_clock::now() - local_start);
    }

    const steady_clock::duration temp_time =  steady_clock::now() - global_start;
    if (verbose) std::cout << "Simulation and result analysis took a total of: " << duration_cast<milliseconds>(temp_time).count() << "[ms] \n";

    r_writer->write_summary(total_simulations,temp_time);
    
    free(total_memory_heap);

    // for (void* it : free_list)
    // {
    //     free(it);
    // }
}
