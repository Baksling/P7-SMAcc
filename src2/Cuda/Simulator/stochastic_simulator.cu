#include "stochastic_simulator.h"
#include "thread_pool.h"
#include <map>
#include "device_launch_parameters.h"
#include <chrono>
#include "simulator_tools.h"
#include "../Domain/edge_t.h"

using namespace std::chrono;



CPU GPU edge_t* find_valid_edge_heap(
    simulator_state* state,
    const lend_array<edge_t*>* edges,
    curandState* r_state)
{
    // return nullptr;
    edge_t** valid_edges = static_cast<edge_t**>(malloc(sizeof(edge_t*) * edges->size()));  // NOLINT(bugprone-sizeof-expression)
    if(valid_edges == nullptr) printf("COULD NOT ALLOCATE HEAP MEMORY\n");
    int valid_count = 0;
    
    for (int i = 0; i < edges->size(); ++i)
    {
        valid_edges[i] = nullptr; //clean malloc
        edge_t* edge = edges->get(i);
        if(edge->evaluate_constraints(state))
            valid_edges[valid_count++] = edge;
    }
    
    if(valid_count == 0)
    {
        free(valid_edges);
        return nullptr;
    }
    if(valid_count == 1)
    {
        edge_t* result = valid_edges[0];
        free(valid_edges);
        return result;
    }

    //summed weight
    float weight_sum = 0.0f;
    for(int i = 0; i < valid_count; i++)
    {
        weight_sum += valid_edges[i]->get_weight();
    }

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const float r_val = (1.0f - curand_uniform(r_state))*weight_sum;
    float r_acc = 0.0; 

    //pick the weighted random value.
    for (int i = 0; i < valid_count; ++i)
    {
        edge_t* temp = valid_edges[i];
        r_acc += temp->get_weight();
        if(r_val < r_acc)
        {
            free(valid_edges);
            return temp;
        }
    }

    //This should be handled in for loop.
    //This is for safety :)
    edge_t* edge = valid_edges[valid_count - 1];
    free(valid_edges);
    return edge;
}

CPU GPU edge_t* find_valid_edge_fast(
    simulator_state* state,
    const lend_array<edge_t*>* edges,
    curandState* r_state)
{
    unsigned long long valid_edges_bitarray = 0UL;
    unsigned int valid_count = 0;
    edge_t* valid_edge = nullptr;
    
    for (int i = 0; i < edges->size(); ++i)
    {
        edge_t* edge = edges->get(i);
        if(edge->evaluate_constraints(state))
        {
            bit_handler::set_bit(&valid_edges_bitarray, i);
            valid_edge = edge;
            valid_count++;
        }
    }
    
    if(valid_count == 0) return nullptr;
    if(valid_count == 1 && valid_edge != nullptr) return valid_edge;

    //summed weight
    float weight_sum = 0.0f;
    for(int i = 0; i  < edges->size(); i++)
    {
        if(bit_handler::bit_is_set(&valid_edges_bitarray, i))
            weight_sum += edges->get(i)->get_weight();
    }

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const float r_val = (1.0f - curand_uniform(r_state)) * weight_sum;
    float r_acc = 0.0; 

    //pick the weighted random value.
    valid_edge = nullptr; //reset valid edge !IMPORTANT
    for (int i = 0; i < edges->size(); ++i)
    {
        if(!bit_handler::bit_is_set(&valid_edges_bitarray, i)) continue;

        valid_edge = edges->get(i);
        r_acc += valid_edge->get_weight();
        if(r_val < r_acc)
        {
            return valid_edge;
        }
    }
    return valid_edge;
}

CPU GPU edge_t* choose_next_edge(simulator_state* state, const lend_array<edge_t*>* edges, curandState* r_state)
{
    //if no possible edges, return null pointer
    if(edges->size() == 0) return nullptr;
    if(edges->size() == 1)
    {
        edge_t* edge = edges->get(0);
        return edge->evaluate_constraints(state)
                ? edge
                : nullptr;
    }

    if(static_cast<unsigned long long>(edges->size()) < sizeof(unsigned long long)*8)
        return find_valid_edge_fast(state, edges, r_state);
    else
        return find_valid_edge_heap(state, edges, r_state);
}

CPU GPU void progress_time(simulator_state* state, const node_t* node, curandState* r_state)
{
    //Get random uniform value between ]0.0f, 0.1f] * difference gives a random uniform range of ]0, diff]
    double max_progression = 0.0; //only set when has_upper_bound == true
    const bool has_upper_bound = node->max_time_progression(&state->timers, &max_progression);
    double time_progression;


    if(has_upper_bound)
    {
        time_progression = (1.0 - curand_uniform_double(r_state)) * max_progression;
    }
    else
    {
        const double lambda = node->get_lambda(state);
        time_progression = (-log(curand_uniform_double(r_state))) / lambda;
    }
    

    //update all timers by adding time_progression to each
    for(int i = 0; i < state->timers.size(); i++)
    {
        state->timers.at(i)->add_time(time_progression);
    }
}



CPU GPU void populate_simulation_result(simulation_result* result,
    const int id, const unsigned steps, const simulator_state* state)
{
    result->id = id;
    result->steps = steps;

    for (int i = 0; i < state->variables.size(); ++i)
    {
        result->variables_max_value[i] = state->variables.at(i)->get_max_value();
    }
}

CPU GPU void simulate_stochastic_model(
    const stochastic_model_t* model,
    const model_options* options,
    curandState* r_state,
    simulation_result* output,
    const unsigned long idx
)
{
    curand_init(options->seed, idx, idx, &r_state[idx]);
    
    array_t<clock_variable> internal_timers = model->create_internal_timers();
    array_t<clock_variable> internal_variables = model->create_internal_variables();

    
    simulator_state state = {
        cuda_stack<double>(options->max_expression_depth),
        cuda_stack<expression*>(options->max_expression_depth*2+1),
        lend_array<clock_variable>(&internal_variables),
        lend_array<clock_variable>(&internal_timers)
    };

    
    for (unsigned int i = 0; i < options->simulation_amount; ++i)
    {
        if(idx == 0 && i % 100 == 0) printf("Progress: %d/%d\n", i, options->simulation_amount);
        //calculate the current simulation id
        const unsigned int sim_id = i + options->simulation_amount * static_cast<unsigned int>(idx);

        populate_simulation_result(&output[sim_id], HIT_MAX_STEPS, 0, &state);
        
        model->reset_timers(&internal_timers, &internal_variables);
        node_t* current_node = model->get_start_node();
        unsigned int steps = 0;
        bool hit_max_steps;
        
        while(true)
        {
            if(steps >= options->max_steps_pr_sim)
            {
                hit_max_steps = true;
                break;
            }
            steps++;
            //check current position is valid
            if(!current_node->evaluate_invariants(&state))
            {
                hit_max_steps = true;
                break;
            }
            
            //Progress time
            if (!current_node->is_branch_point())
            {
                progress_time(&state, current_node, &r_state[idx]);
            }
            
            const lend_array<edge_t*> outgoing_edges = current_node->get_edges();
            if(outgoing_edges.size() <= 0)
            {
                hit_max_steps = false;
                break;
            }
            
            const edge_t* next_edge = choose_next_edge(&state, &outgoing_edges, &r_state[idx]);
            if(next_edge == nullptr)
            {
                continue;
            }
            current_node = next_edge->get_dest();
            next_edge->execute_updates(&state);
            if(current_node->is_goal_node())
            {
                hit_max_steps = false;
                break;
            }
        }
        
        if (hit_max_steps)
        {
            populate_simulation_result(&output[sim_id], HIT_MAX_STEPS, steps, &state);
        }
        else
        {
            if(current_node->is_goal_node())
                populate_simulation_result(&output[sim_id], current_node->get_id(), steps, &state);
            else
                populate_simulation_result(&output[sim_id], HIT_MAX_STEPS, steps, &state);

        }
    }

    internal_timers.free_array();
    internal_variables.free_array();
    state.expression_stack.free_internal();
    state.value_stack.free_internal();
    if(idx == 0) printf("Progress: %d/%d\n", options->simulation_amount, options->simulation_amount);
}


__global__ void gpu_simulate(
    const stochastic_model_t* model,
    const model_options* options,
    curandState* r_state,
    simulation_result* output
    )
{
    const unsigned long idx = threadIdx.x + blockDim.x * blockIdx.x;
    simulate_stochastic_model(model, options, r_state, output, idx);
}




void stochastic_simulator::simulate_cpu(stochastic_model_t* model, simulation_strategy* strategy)
{
    //setup start variables
    const unsigned long total_simulations = strategy->total_simulations();
    //setup random state
    curandState* state = static_cast<curandState*>(malloc(sizeof(curandState) * strategy->degree_of_parallelism()));
    
    //setup results array
    const unsigned int variable_count = model->get_variable_count();
    std::list<void*> free_list;
    std::unordered_map<node_t*, node_t*> node_map;
    std::map<int, node_result> node_results;
    const allocation_helper allocator = { &free_list, &node_map };
    
    array_t<variable_result> variable_r = result_handler::allocate_variable_results(variable_count);
    const lend_array<variable_result> lend_variable_r = lend_array<variable_result>(&variable_r);
    simulation_result* sim_results = result_handler::allocate_results(strategy, variable_count, &allocator, false);

    //setup simulation options
    const model_options options = {
        strategy->simulation_amounts,
        strategy->max_sim_steps,
        static_cast<unsigned long>(time(nullptr)),
        strategy->max_time_progression,
        500
    };

    const steady_clock::time_point start = steady_clock::now();
    std::cout << "Started running!\n";
    thread_pool pool(strategy->cpu_threads_n);

    for (int i = 0; i < strategy->degree_of_parallelism(); i++)
    {
        pool.queue_job([model, options, state, sim_results, i]()
        {
            simulate_stochastic_model(model, &options, state, sim_results, i);
        });
    }
    pool.start();


    while(pool.is_busy())
    {
    }
    pool.stop();

    std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "[ms] \n";
    std::cout << "Reading results...\n";
    result_handler::read_results(sim_results, total_simulations, &node_results, &lend_variable_r);
    result_handler::print_results(&node_results, &lend_variable_r, total_simulations);

    variable_r.free_array();
    free(sim_results);
    free(state);
}

void stochastic_simulator::simulate_gpu(const stochastic_model_t* model, simulation_strategy* strategy)
{
     //setup start variables
    const unsigned long total_simulations = strategy->total_simulations();
    //setup random state
    curandState* state;
    cudaMalloc(&state, sizeof(curandState) * strategy->degree_of_parallelism());
    
    //setup results array
    const unsigned int variable_count = model->get_variable_count();
    std::list<void*> free_list;
    std::unordered_map<node_t*, node_t*> node_map;
    std::map<int, node_result> node_results;
    const allocation_helper allocator = { &free_list, &node_map };
    array_t<variable_result> variable_r = result_handler::allocate_variable_results(variable_count);
    const lend_array<variable_result> lend_variable_r = lend_array<variable_result>(&variable_r);
    simulation_result* sim_results = result_handler::allocate_results(strategy, variable_count, &allocator, true);

    //allocate model to cuda
    stochastic_model_t* model_d = nullptr;
    model->cuda_allocate(&model_d, &allocator);
    
    //implement here
    model_options* options_d = nullptr;
    const model_options options = {
        strategy->simulation_amounts,
        strategy->max_sim_steps,
        static_cast<unsigned long>(time(nullptr)),
        strategy->max_time_progression,
        500
    };
    cudaMalloc(&options_d, sizeof(model_options));
    cudaMemcpy(options_d, &options, sizeof(model_options), cudaMemcpyHostToDevice);


    //run simulations
    std::cout << "Started running!\n";
    if(cudaSuccess != cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8589934592))
        printf("COULD NOT CHANGE LIMIT");
    
    const steady_clock::time_point start = steady_clock::now();

    
    
    for (int i = 0; i < strategy->sim_count; ++i)
    {
        //simulate on device
        gpu_simulate<<<strategy->block_n, strategy->threads_n>>>(model_d, options_d, state, sim_results);
        
        //wait for all processes to finish
        cudaDeviceSynchronize();
        printf("I do the reee1");
        if(cudaPeekAtLastError() != cudaSuccess) break;

        printf("I do the REEEEEEEEEEE");
        //count result unless last sim
        if(i < strategy->sim_count) 
        {
            std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "[ms] \n";
            std::cout << "Reading results...\n";
            simulation_result* local_results = static_cast<simulation_result*>(malloc(sizeof(simulation_result)*total_simulations));
            cudaMemcpy(local_results, sim_results, sizeof(simulation_result)*total_simulations, cudaMemcpyDeviceToHost);
            result_handler::read_results(local_results, total_simulations, &node_results, &lend_variable_r);
            free(local_results);
        }
    }

    if(strategy->sim_count > 1)
    {
        std::cout << "Total Simulation time: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "[ms] \n";
    }

    const cudaError status = cudaPeekAtLastError();
    if(cudaPeekAtLastError() == cudaSuccess)
    {
        result_handler::print_results(&node_results, &lend_variable_r, total_simulations * strategy->sim_count);
    }
    else
    {
        printf("An error occured during device execution" );
        printf("CUDA error code: %s\n", cudaGetErrorString(status));
        exit(status);  // NOLINT(concurrency-mt-unsafe)
        return;
    }

    variable_r.free_array();
    cudaFree(sim_results);
    cudaFree(state);
    cudaFree(options_d);
    
    for (void* it : free_list)
    {
        cudaFree(it);
    }
}
