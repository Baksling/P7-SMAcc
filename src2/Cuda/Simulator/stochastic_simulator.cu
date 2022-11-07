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

model_options stochastic_simulator::build_options(const stochastic_model_t* model,
    const simulation_strategy* strategy)
{
    const domain_analysis_visitor analyser = domain_analysis_visitor();
    //analyser.visit(model);
    
    return model_options{
        strategy->simulations_per_thread,
        static_cast<unsigned long>(time(nullptr)),
        // analyser.get_max_expression_depth()
        strategy->use_max_steps,
        strategy->max_sim_steps,
        strategy->max_time_progression,

        500,
        model->get_models_count(),
        model->get_variable_count(),
        model->get_timer_count()
    };
}

void stochastic_simulator::simulate_gpu(const stochastic_model_t* model, const simulation_strategy* strategy, const result_writer* r_writer)
{
     //setup start variables
    const unsigned long total_simulations = strategy->total_simulations();

    //setup results array
    const unsigned int variable_count = model->get_variable_count();
    const unsigned int model_count = model->get_models_count();
    
    std::list<void*> free_list;
    std::unordered_map<node_t*, node_t*> node_map;
    const allocation_helper allocator = { &free_list, &node_map };

    //TODO slim this down, this is handled by the result writer
    // std::map<int, node_result> result_map;
    // array_t<variable_result> variable_r = simulator_tools::allocate_variable_results(variable_count);
    // const lend_array<variable_result> lend_variable_r = lend_array<variable_result>(&variable_r);
    // simulation_result* sim_results = simulator_tools::allocate_results(strategy, variable_count, model_count, &free_list, true);

     const simulation_result_container results = simulation_result_container(
        total_simulations,
        model_count,
        variable_count,
        true);
    simulation_result_container* d_result = results.cuda_allocate(&allocator);
    
    
    //Allocate model on cuda
    stochastic_model_t* model_d = nullptr;
    cudaMalloc(&model_d, sizeof(stochastic_model_t));
    allocator.free_list->push_back(model_d);
    model->cuda_allocate(model_d, &allocator);

    //implement here
    model_options* options_d = nullptr;
    const model_options options = build_options(model, strategy);
    cudaMalloc(&options_d, sizeof(model_options));
    cudaMemcpy(options_d, &options, sizeof(model_options), cudaMemcpyHostToDevice);

    const unsigned long long int thread_memory_size = options.get_cache_size();
    void* total_memory_heap;
    cudaMalloc(&total_memory_heap, thread_memory_size * strategy->degree_of_parallelism());
    free_list.push_back(total_memory_heap);
    
    printf("Allocating: %llu bits\n", thread_memory_size * strategy->degree_of_parallelism());

    //run simulations
    std::cout << "Started running!\n";
    const steady_clock::time_point global_start = steady_clock::now();
    
    for (unsigned i = 0; i < strategy->simulation_runs; ++i)
    {
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
        std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - local_start).count() << "[ms] \n";
        std::cout << "Reading results...\n";
        
        //simulator_tools::read_results(local_results, total_simulations, model_count, &result_map, &lend_variable_r, true);
        r_writer->write_results(&results, total_simulations, variable_count, steady_clock::now() - local_start, true);
    }

    std::cout << "Simulation and result analysis took a total of: " << duration_cast<milliseconds>(steady_clock::now() - global_start).count() << "[ms] \n";

    //simulator_tools::print_results(&result_map, &lend_variable_r, total_simulations);
    
    //free local variabels
    cudaFree(options_d);
    results.free_internals();
    for (void* it : free_list)
    {
        cudaFree(it);
    }
}

void stochastic_simulator::simulate_cpu(
    const stochastic_model_t* model,
    const simulation_strategy* strategy,
    const result_writer* r_writer)
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
    
    const model_options options = build_options(model, strategy);

    const unsigned long long int thread_memory_size = options.get_cache_size();
    void* total_memory_heap = malloc(thread_memory_size * strategy->degree_of_parallelism());

    std::cout << "Started running!\n";
    const steady_clock::time_point global_start = steady_clock::now();
    
    for (unsigned i = 0; i < strategy->simulation_runs; ++i)
    {
        const steady_clock::time_point local_start = steady_clock::now();
        const bool success = stochastic_engine::run_cpu(model, &options, &results, strategy, total_memory_heap);
        if(!success)
        {
            printf("An unsuccessful CPU simulation has occured. Stopping simulation.");
            break;
        }
    
        std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - local_start).count() << "[ms] \n";
        std::cout << "Reading results...\n";
        r_writer->write_results(&results, total_simulations, variable_count, steady_clock::now() - local_start, false);
    }
    
    std::cout << "Simulation and result analysis took a total of: " << duration_cast<milliseconds>(steady_clock::now() - global_start).count() << "[ms] \n";
    
    // for (void* it : free_list)
    // {
    //     free(it);
    // }
}
