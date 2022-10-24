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
        strategy->simulation_amounts,
        strategy->max_sim_steps,
        static_cast<unsigned long>(time(nullptr)),
        100,
        // analyser.get_max_expression_depth()
        500
    };
}

void stochastic_simulator::simulate_gpu(const stochastic_model_t* model, const simulation_strategy* strategy, const result_writer* r_writer)
{
     //setup start variables
    const unsigned long total_simulations = strategy->total_simulations();

    //setup results array
    const unsigned int variable_count = model->get_variable_count();
    std::list<void*> free_list;
    std::unordered_map<node_t*, node_t*> node_map;
    const allocation_helper allocator = { &free_list, &node_map };

    //TODO slim this down, this is handled by the result writer
    std::map<int, node_result> result_map;
    array_t<variable_result> variable_r = simulator_tools::allocate_variable_results(variable_count);
    const lend_array<variable_result> lend_variable_r = lend_array<variable_result>(&variable_r);
    simulation_result* sim_results = simulator_tools::allocate_results(strategy, variable_count, &free_list, true);

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

    //run simulations
    std::cout << "Started running!\n";
    const steady_clock::time_point global_start = steady_clock::now();
    
    for (unsigned i = 0; i < strategy->sim_count; ++i)
    {
        //start time local
        const steady_clock::time_point local_start = steady_clock::now();
        
        //simulate on device
        const bool success = stochastic_engine::run_gpu(model_d, options_d, sim_results, strategy);
        if(!success)
        {
            printf("An unsuccessful GPU simulation has occured. Stopping simulation.");
            break;
        }
    
        //count result unless last sim
        std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - local_start).count() << "[ms] \n";
        std::cout << "Reading results...\n";
        simulation_result* local_results = static_cast<simulation_result*>(malloc(sizeof(simulation_result)*total_simulations));
        cudaMemcpy(local_results, sim_results, sizeof(simulation_result)*total_simulations, cudaMemcpyDeviceToHost);
        simulator_tools::read_results(local_results, total_simulations, &result_map, &lend_variable_r, true);
        free(local_results);
    }

    std::cout << "Simulation and result analysis took a total of: " << duration_cast<milliseconds>(steady_clock::now() - global_start).count() << "[ms] \n";

    simulator_tools::print_results(&result_map, &lend_variable_r, total_simulations);
    
    //free local variabels
    variable_r.free_array();
    cudaFree(sim_results);
    cudaFree(options_d);
    for (void* it : free_list)
    {
        cudaFree(it);
    }
}

void stochastic_simulator::simulate_cpu(const stochastic_model_t* model, const simulation_strategy* strategy)
{
    //setup start variables
    const unsigned long total_simulations = strategy->total_simulations();

    //setup results array
    const unsigned int variable_count = model->get_variable_count();

    //TODO slim this down, this is handled by the result writer
    std::list<void*> free_list;
    std::map<int, node_result> result_map;
    array_t<variable_result> variable_r = simulator_tools::allocate_variable_results(variable_count);
    const lend_array<variable_result> lend_variable_r = lend_array<variable_result>(&variable_r);
    simulation_result* sim_results = simulator_tools::allocate_results(strategy, variable_count, &free_list, false);

    const model_options options = build_options(model, strategy);

    std::cout << "Started running!\n";
    const steady_clock::time_point global_start = steady_clock::now();

    for (unsigned i = 0; i < strategy->sim_count; ++i)
    {
        const steady_clock::time_point local_start = steady_clock::now();
        const bool success = stochastic_engine::run_cpu(model, &options, sim_results, strategy);
        if(!success)
        {
            printf("An unsuccessful CPU simulation has occured. Stopping simulation.");
            break;
        }

        std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - local_start).count() << "[ms] \n";
        std::cout << "Reading results...\n";
        simulator_tools::read_results(sim_results, total_simulations, &result_map, &lend_variable_r, true);
    }

    std::cout << "Simulation and result analysis took a total of: " << duration_cast<milliseconds>(steady_clock::now() - global_start).count() << "[ms] \n";


    free(sim_results);
    variable_r.free_array();
    for (void* it : free_list)
    {
        free(it);
    }
}
