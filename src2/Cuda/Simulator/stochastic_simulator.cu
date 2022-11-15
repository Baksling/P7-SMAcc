#include "stochastic_simulator.h"
#include "thread_pool.h"
#include "device_launch_parameters.h"
#include <chrono>
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


void print_model_memory_diagnosis(const allocation_helper* helper)
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    printf("Total model memory utilization: %llu bytes (%lf%%).\n",
        helper->allocated_size / 8UL,
        (static_cast<double>(helper->allocated_size) / static_cast<double>(total_mem))*100
    );
}

void print_cache_memory_diagnosis(const size_t total_cache)
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    printf("Total simulation cache memory utilization: %llu bytes (%lf%%).\n",
        static_cast<unsigned long long>(total_cache / 8UL),
        (static_cast<double>(total_cache) / static_cast<double>(total_mem))*100
    );
}


void stochastic_simulator::simulate_gpu(stochastic_model_t* model, const simulation_strategy* strategy, result_writer* r_writer, const bool verbose)
{
     //setup start variables
    const unsigned long total_simulations = strategy->total_simulations();
    allocation_helper allocator = allocation_helper(true);
    
    //Allocate model on cuda
    stochastic_model_t* model_d = nullptr;
    allocator.allocate(&model_d, sizeof(stochastic_model_t));
    model->cuda_allocate(model_d, &allocator);
    
    //move options to GPU
    model_options options = build_options(model, strategy);
    model_options* options_d = nullptr;
    allocator.allocate(&options_d, sizeof(model_options));
    cudaMemcpy(options_d, &options, sizeof(model_options), cudaMemcpyHostToDevice);

    if(verbose) print_model_memory_diagnosis(&allocator);
    
    //allocate simulation cache.
    const unsigned long long int thread_memory_size = options.get_cache_size() * strategy->degree_of_parallelism();
    void* total_memory_heap = nullptr;
    allocator.allocate(&total_memory_heap, thread_memory_size);

    if(verbose) print_cache_memory_diagnosis(thread_memory_size);
    
    result_manager* results;
    result_manager* d_results;

    const size_t pre_results_mem_allocated = allocator.allocated_size;
    result_manager::init_unified(&results, &d_results, model, strategy, &allocator);
    
    if(verbose) results->print_memory_usage(&allocator, pre_results_mem_allocated);
    
    //run simulations
    if (verbose) std::cout << "\nStarted running!\n";
    const steady_clock::time_point global_start = steady_clock::now();
    
    for (unsigned i = 0; i < strategy->simulation_runs; ++i)
    {
        options.seed = static_cast<unsigned long>(time(nullptr));

        //start time local
        const steady_clock::time_point local_start = steady_clock::now();
        
        //simulate on device
        const bool success = stochastic_engine::run_gpu(model_d, options_d, d_results, strategy, total_memory_heap);
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
        r_writer->write(results, steady_clock::now() - local_start);
        results->clear();
    }

     const steady_clock::duration temp_time = steady_clock::now() - global_start;
    
    if (verbose) std::cout << "Simulation and result analysis took a total of: " << duration_cast<milliseconds>(temp_time).count() << "[ms] \n";
    r_writer->write_summary(total_simulations,temp_time);
    
    //simulator_tools::print_results(&result_map, &lend_variable_r, total_simulations);
    
    //free local variables
    free(results);
    allocator.free_allocations();
}

void stochastic_simulator::simulate_cpu(
    stochastic_model_t* model,
    const simulation_strategy* strategy,
    result_writer* r_writer,
    const bool verbose)
{
    //setup start variables

    
    const unsigned long total_simulations = strategy->total_simulations();

    allocation_helper allocator = allocation_helper(false);
    model_options options = build_options(model, strategy);

    const unsigned long long int thread_memory_size = options.get_cache_size()  * strategy->degree_of_parallelism();
    void* total_memory_heap = nullptr;
    allocator.allocate(&total_memory_heap, thread_memory_size);

    //Allocate results
    result_manager* results = result_manager::init(model, strategy, &allocator);

    if (verbose) std::cout << "\nStarted running!\n";
    const steady_clock::time_point global_start = steady_clock::now();
    
    for (unsigned i = 0; i < strategy->simulation_runs; ++i)
    {
        options.seed = static_cast<unsigned long>(time(nullptr));
        const steady_clock::time_point local_start = steady_clock::now();
        const bool success = stochastic_engine::run_cpu(model, &options, results, strategy, total_memory_heap);
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
        r_writer->write(results, steady_clock::now() - local_start);
        results->clear();
    }

    const steady_clock::duration temp_time =  steady_clock::now() - global_start;
    if (verbose) std::cout << "Simulation and result analysis took a total of: " << duration_cast<milliseconds>(temp_time).count() << "[ms] \n";
    
    r_writer->write_summary(total_simulations,temp_time);

    allocator.free_allocations();
}
