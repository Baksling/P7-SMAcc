#include "CudaSimulator.h"

#include <map>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>


#define HIT_MAX_STEPS (-1)
using namespace std::chrono;


GPU edge_t* choose_next_edge(const lend_array<edge_t*>* edges, const lend_array<clock_timer_t>* timer_arr,
    curandState* states, const unsigned int thread_id)
{
    //if no possible edges, return null pointer
    if(edges->size() == 0) return nullptr;
    if(edges->size() == 1)
    {
        edge_t* edge = edges->get(0);
        return edge->evaluate_constraints(timer_arr)
                ? edge
                : nullptr;
    }

    // return nullptr;
    edge_t** valid_edges = static_cast<edge_t**>(malloc(sizeof(void*) * edges->size()));
    if(valid_edges == nullptr) printf("COULD NOT ALLOCATE HEAP MEMORY");
    int valid_count = 0;
    
    for (int i = 0; i < edges->size(); ++i)
    {
        valid_edges[i] = nullptr; //clean malloc
        edge_t* edge = edges->get(i);
        if(edge->evaluate_constraints(timer_arr))
        {
            valid_edges[valid_count] = edge;
            valid_count++;
        }
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
        const edge_t* temp = (valid_edges[i]);
        weight_sum += temp->get_weight();
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

GPU void progress_time(const lend_array<clock_timer_t>* timers, const double difference, curandState* states, const unsigned int thread_id)
{
    //Get random uniform value between ]0.0f, 0.1f] * difference gives a random uniform range of ]0, diff]
    const double time_progression = difference * curand_uniform_double(&states[thread_id]);

    //update all timers by adding time_progression to each
    for(int i = 0; i < timers->size(); i++)
    {
        timers->at(i)->add_time(time_progression);
    }
}

__global__ void simulate_gpu(
    stochastic_model_t* model,
    const model_options* options,
    curandState* r_state,
    int* output
)
{
    const unsigned long idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(options->seed, idx, idx, &r_state[idx]);

    array_t<clock_timer_t> internal_timers = model->create_internal_timers();
    const lend_array<clock_timer_t> lend_internal_timers = lend_array<clock_timer_t>(&internal_timers);

    if(idx == 0) printf("Progress: %d/%d\n", 0, options->simulation_amount);
    for (unsigned int i = 0; i < options->simulation_amount; ++i)
    {
        if(idx == 0 && i+1 % 100 == 0) printf("Progress: %d/%d\n", i+1, options->simulation_amount);
        //calculate the current simulation id
        const unsigned int sim_id = i + options->simulation_amount * static_cast<unsigned int>(idx);
        
        output[sim_id] = HIT_MAX_STEPS;
        model->reset_timers(&internal_timers);
        node_t* current_node = model->get_start_node();
        unsigned int steps = 0;
        bool hit_max_steps = false;
        
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
                const double max_progression = current_node->max_time_progression(&lend_internal_timers);
                printf("diff: %f\n", max_progression);
                progress_time(&lend_internal_timers, max_progression, r_state, idx);
            }

            const lend_array<edge_t*> outgoing_edges = current_node->get_edges();
            if(outgoing_edges.size() <= 0)
            {
                break;
            }

            edge_t* next_edge = choose_next_edge(&outgoing_edges, &lend_internal_timers, r_state, idx);
            if(next_edge == nullptr)
            {
                continue;
            }

            current_node = next_edge->get_dest();
            next_edge->execute_updates(&lend_internal_timers);
            if(current_node->is_goal_node())
            {
                break;
            }
        }
        
        if (hit_max_steps)
        {
            output[sim_id] = HIT_MAX_STEPS;
        }
        else
        {
            output[sim_id] = current_node->get_id();
        }
    }

    internal_timers.free_array();
}

void read_results(const int* cuda_results, const int total_simulations, std::map<int, unsigned long>* results)
{
    int* local_results = static_cast<int*>(malloc(sizeof(int)*total_simulations));
    cudaMemcpy(local_results, cuda_results, sizeof(int)*total_simulations, cudaMemcpyDeviceToHost);

    for (int i = 0; i < total_simulations; ++i)
    {
        int id = local_results[i];
        int count = 0;
        if(results->count(id) == 1)
        {
            count = (*results)[id];
        }

        results->insert_or_assign(id, count+1);

    }
    free(local_results);
}

float calc_percentage(const unsigned long counter, const unsigned long divisor)
{
    return (static_cast<float>(counter)/static_cast<float>(divisor))*100;
} 

void print_results(std::map<int,unsigned long>* result_map, const unsigned long result_size)
{
    std::cout << "\n";
    for (const std::pair<const int, int> it : (*result_map))
    {
        if(it.first == HIT_MAX_STEPS) continue;
        const float percentage = calc_percentage(it.second, result_size);
        std::cout << "Node: " << it.first << " reached " << it.second << " times. (" << percentage << ")%\n";
    }
    const float percentage = calc_percentage((*result_map)[HIT_MAX_STEPS], result_size);
    std::cout << "No goal state was reached " << (*result_map)[HIT_MAX_STEPS] << " times. (" << percentage << ")%\n";
    std::cout << "\n";
    std::cout << "Nr of simulations: " << result_size << "\n";
    std::cout << "\n";

}

void cuda_simulator::simulate(stochastic_model_t* model, simulation_strategy* strategy)
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
    const steady_clock::time_point start = steady_clock::now();
    std::cout << "Started running!\n";
    if(cudaSuccess != cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8589934592))
        printf("COULD NOT CHANGE LIMIT");
    
    for (int i = 0; i < strategy->sim_count; ++i)
    {
        //simulate on device
        simulate_gpu<<<strategy->block_n, strategy->threads_n>>>(
        model_d, options_d, state, cuda_results);

        //wait for all processes to finish
        cudaDeviceSynchronize();
        if(cudaPeekAtLastError() != cudaSuccess) break;

        //count result unless last sim
        if(i < strategy->sim_count - 1) 
        {
            read_results(cuda_results, total_simulations, &node_results);
        }
    }
    
    std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "[ms] \n";

    const cudaError status = cudaPeekAtLastError();
    if(cudaPeekAtLastError() == cudaSuccess)
    {
        std::cout << "Reading results...\n";
        read_results(cuda_results, total_simulations, &node_results);
        print_results(&node_results, total_simulations * strategy->sim_count);
    }
    else
    {
        printf("An error occured during device execution" );
        printf("CUDA error code: %d\n", status);
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
