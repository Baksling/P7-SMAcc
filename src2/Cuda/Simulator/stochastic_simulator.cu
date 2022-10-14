#include "stochastic_simulator.h"
using namespace std::chrono;


CPU GPU edge_t* find_valid_edge_heap(const lend_array<edge_t*>* edges,
    const lend_array<clock_timer_t>* timer_arr, const lend_array<system_variable>* variables,
    curandState* states, const unsigned int thread_id)
{
    // return nullptr;
    edge_t** valid_edges = static_cast<edge_t**>(malloc(sizeof(edge_t*) * edges->size()));  // NOLINT(bugprone-sizeof-expression)
    if(valid_edges == nullptr) printf("COULD NOT ALLOCATE HEAP MEMORY\n");
    int valid_count = 0;
    
    for (int i = 0; i < edges->size(); ++i)
    {
        valid_edges[i] = nullptr; //clean malloc
        edge_t* edge = edges->get(i);
        if(edge->evaluate_constraints(timer_arr, variables))
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
    const float r_val = (1.0f - curand_uniform(&states[thread_id]))*weight_sum;
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

CPU GPU edge_t* find_valid_edge_fast(const lend_array<edge_t*>* edges,
    const lend_array<clock_timer_t>* timers, const lend_array<system_variable>* variables,
    curandState* states, const unsigned int thread_id)
{
    unsigned long long valid_edges_bitarray = 0UL;
    unsigned int valid_count = 0;
    edge_t* valid_edge = nullptr;
    
    for (int i = 0; i < edges->size(); ++i)
    {
        edge_t* edge = edges->get(i);
        if(edge->evaluate_constraints(timers,variables))
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
    const float r_val = (1.0f - curand_uniform(&states[thread_id])) * weight_sum;
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

CPU GPU edge_t* choose_next_edge(const lend_array<edge_t*>* edges,
    const lend_array<clock_timer_t>* timer_arr, const lend_array<system_variable>* variables,
    curandState* states, const unsigned int thread_id)
{
    //if no possible edges, return null pointer
    if(edges->size() == 0) return nullptr;
    if(edges->size() == 1)
    {
        edge_t* edge = edges->get(0);
        return edge->evaluate_constraints(timer_arr, variables)
                ? edge
                : nullptr;
    }

    if(static_cast<unsigned long long>(edges->size()) < sizeof(unsigned long long)*8)
        return find_valid_edge_fast(edges, timer_arr, variables, states, thread_id);
    else
        return find_valid_edge_heap(edges, timer_arr, variables, states, thread_id);
}

CPU GPU void progress_time(const lend_array<clock_timer_t>* timers, const node_t* node,
    const double max_global_progress, curandState* r_state)
{
    //Get random uniform value between ]0.0f, 0.1f] * difference gives a random uniform range of ]0, diff]
    double max_progression = 0.0; //only set when has_upper_bound == true
    const bool has_upper_bound = node->max_time_progression(timers, &max_progression);
    const double lambda = static_cast<double>(node->get_lambda());
    double time_progression;
    
    if(lambda <= 0 && has_upper_bound) //chose uniform distribution between 0 and upper bound
    {
        time_progression = (1.0 - curand_uniform_double(r_state)) * max_progression;
    }
    else if(lambda > 0 && has_upper_bound) //choose exponential distribution between 0 and upper bound
    {
        // time_progression = (max_progression
        // - ( (1 - lambda*exp(-lambda* curand_uniform_double(r_state) * max_progression))
        // * max_progression)) / lambda;
        
        // time_progression = (exp(-lambda*0) - exp(-lambda*curand_uniform_double(r_state) * max_progression))
        //                 /  (exp(-lambda*0) - exp(-lambda*max_progression));
        
        time_progression = (1 - exp(-lambda*curand_uniform_double(r_state) * max_progression))
                        /  (1 - exp(-lambda*max_progression));
    }
    else if(lambda > 0) //choose exponential distribution between 0 and infinity
    {
        time_progression = (-log(curand_uniform_double(r_state))) / lambda;
    }
    else //choose uniform distribution between 0 and global max
    {
        time_progression = (1.0 - curand_uniform_double(r_state)) * max_global_progress;
    }
    
    
    if(has_upper_bound && time_progression > max_progression)
    {
        printf("Adjusted clock bound, %lf to %lf\n", time_progression, max_progression);
        time_progression = max_progression;

    }

    //update all timers by adding time_progression to each
    for(int i = 0; i < timers->size(); i++)
    {
        timers->at(i)->add_time(time_progression);
    }
}


CPU GPU void simulate_stochastic_model(
    const stochastic_model_t* model,
    const model_options* options,
    curandState* r_state,
    int* output,
    const unsigned long idx,
    const double max_global_progression
)
{
    curand_init(options->seed, idx, idx, &r_state[idx]);

    array_t<clock_timer_t> internal_timers = model->create_internal_timers();
    array_t<system_variable> internal_variables = model->create_internal_variables();
    const lend_array<clock_timer_t> lend_internal_timers = lend_array<clock_timer_t>(&internal_timers);
    const lend_array<system_variable> lend_internal_variables = lend_array<system_variable>(&internal_variables);

    for (unsigned int i = 0; i < options->simulation_amount; ++i)
    {
        if(idx == 0 && i % 100 == 0) printf("Progress: %d/%d\n", i, options->simulation_amount);
        //calculate the current simulation id
        const unsigned int sim_id = i + options->simulation_amount * static_cast<unsigned int>(idx);
        
        output[sim_id] = HIT_MAX_STEPS;
        model->reset_timers(&internal_timers);
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
            if(!current_node->evaluate_invariants(&lend_internal_timers))
            {
                hit_max_steps = true;
                break;
            }
            //Progress time
            if (!current_node->is_branch_point())
            {
                progress_time(&lend_internal_timers, current_node, max_global_progression,  &r_state[idx]);
            }
            const lend_array<edge_t*> outgoing_edges = current_node->get_edges();
            if(outgoing_edges.size() <= 0)
            {
                hit_max_steps = false;
                break;
            }
            const edge_t* next_edge = choose_next_edge(&outgoing_edges,
                &lend_internal_timers, &lend_internal_variables, r_state, idx);
            if(next_edge == nullptr)
            {
                continue;
            }
            current_node = next_edge->get_dest();
            next_edge->execute_updates(&lend_internal_timers, &lend_internal_variables);
            if(current_node->is_goal_node())
            {
                hit_max_steps = false;
                break;
            }
        }
        
        if (hit_max_steps)
        {
            output[sim_id] = HIT_MAX_STEPS;
        }
        else
        {
            if(current_node->is_goal_node())
                output[sim_id] = current_node->get_id();
            else
                output[sim_id] = HIT_MAX_STEPS;
        }
    }

    internal_timers.free_array();
    if(idx == 0) printf("Progress: %d/%d\n", options->simulation_amount, options->simulation_amount);
}


__global__ void gpu_simulate(
    const stochastic_model_t* model,
    const model_options* options,
    curandState* r_state,
    int* output,
    double max_time_progression
)
{
    const unsigned long idx = threadIdx.x + blockDim.x * blockIdx.x;
    simulate_stochastic_model(model, options, r_state, output, idx, max_time_progression);
}




void stochastic_simulator::simulate_cpu(stochastic_model_t* model, simulation_strategy* strategy)
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
    thread_pool pool(strategy->cpu_threads_n);

    for (int i = 0; i < strategy->degree_of_parallelism(); i++)
    {
        pool.queue_job([model, options, state, sim_results, i, strategy]()
        {
            simulate_stochastic_model(model, &options, state, sim_results, i, strategy->max_time_progression);
        });
    }
    pool.start();


    while(pool.is_busy())
    {
    }
    pool.stop();

    std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "[ms] \n";
    std::cout << "Reading results...\n";
    result_handler::read_results(sim_results, total_simulations, &node_results);
    result_handler::print_results(&node_results, total_simulations * strategy->sim_count);

    free(sim_results);
    free(state);
}

void stochastic_simulator::simulate_gpu(stochastic_model_t* model, simulation_strategy* strategy)
{
     //setup start variables
    const unsigned long total_simulations = strategy->total_simulations();
    //setup random state
    curandState* state;
    cudaMalloc(&state, sizeof(curandState) * strategy->degree_of_parallelism());
    
    //setup results array
    const unsigned long size = sizeof(int)*total_simulations;
    int* cuda_results = nullptr;
    const auto r = cudaMalloc(&cuda_results, sizeof(int)*total_simulations);
    printf("allocated %lu (%lu*%lu) bytes successfully: %s\n" ,
        size, static_cast<unsigned long>(sizeof(int)), total_simulations, (r == cudaSuccess ? "True" : "False") );

    //prepare allocation helper
    std::list<void*> free_list;
    std::unordered_map<node_t*, node_t*> node_map;
    const allocation_helper allocator = { &free_list, &node_map };

    //allocate model to cuda
    stochastic_model_t* model_d = nullptr;
    model->cuda_allocate(&model_d, &allocator);
    
    //implement here
    model_options* options_d = nullptr;
    const model_options options = {
        strategy->simulation_amounts,
        strategy->max_sim_steps,
        static_cast<unsigned long>(time(nullptr))
    };
    cudaMalloc(&options_d, sizeof(model_options));
    cudaMemcpy(options_d, &options, sizeof(model_options), cudaMemcpyHostToDevice);


    //run simulations
    std::map<int, unsigned long> node_results;
    std::cout << "Started running!\n";
    if(cudaSuccess != cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8589934592))
        printf("COULD NOT CHANGE LIMIT");
    const steady_clock::time_point start = steady_clock::now();
    
    for (int i = 0; i < strategy->sim_count; ++i)
    {
        //simulate on device
        gpu_simulate<<<strategy->block_n, strategy->threads_n>>>(model_d, options_d, state, cuda_results, strategy->max_time_progression);

        //wait for all processes to finish
        cudaDeviceSynchronize();
        if(cudaPeekAtLastError() != cudaSuccess) break;

        //count result unless last sim
        if(i < strategy->sim_count) 
        {
            std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "[ms] \n";
            std::cout << "Reading results...\n";
            int* local_results = static_cast<int*>(malloc(sizeof(int)*total_simulations));
            cudaMemcpy(local_results, cuda_results, sizeof(int)*total_simulations, cudaMemcpyDeviceToHost);
            result_handler::read_results(local_results, total_simulations, &node_results);
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
        result_handler::print_results(&node_results, total_simulations * strategy->sim_count);
    }
    else
    {
        printf("An error occured during device execution" );
        printf("CUDA error code: %s\n", cudaGetErrorString(status));
        exit(status);  // NOLINT(concurrency-mt-unsafe)
        return;
    }
    
    cudaFree(cuda_results);
    cudaFree(state);
    cudaFree(options_d);
    
    for (void* it : free_list)
    {
        cudaFree(it);
    }
}
