#include "CudaSimulator.h"

#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>


#define NOT_GOAL_STATE (-1)
using namespace std::chrono;


GPU edge_t* choose_next_edge(const lend_array<edge_t>* edges,
    curandState* states, const unsigned int thread_id)
{
    //TODO FILTER TO ONLY VALID EDGES!
    
    //if no possible edges, return null pointer
    if(edges->size() == 0) return nullptr;

    //summed weight
    float weight_sum = 0.0f;
    for(int i = 0; i < edges->size(); i++)
        weight_sum += edges->at(i)->get_weight();

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const float r_val = (1.0f - curand_uniform(&states[thread_id]))*weight_sum;
    float r_acc = 0.0; 

    //pick the weighted random value.
    for (int i = 0; i < edges->size(); ++i)
    {
        r_acc += edges->at(i)->get_weight();
        if(r_val < r_acc) return edges->at(i);
    }

    //This should be handled in for loop.
    //This is for safety :)
    return edges->at(edges->size() - 1);
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
    const stochastic_model_t* model,
    const model_options* options,
    curandState* r_state,
    int* output
)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(options->seed, idx, idx, &r_state[idx]);

    array_t<clock_timer_t> internal_timers = model->create_internal_timers();
    const lend_array<clock_timer_t> lend_internal_timers = lend_array<clock_timer_t>(&internal_timers);

    for (int i = 0; i < options->simulation_amount; ++i)
    {
        //calculate the current simulation id
        const int sim_id = i + options->simulation_amount * static_cast<int>(idx);
        output[sim_id] = NOT_GOAL_STATE;

        model->reset_timers(&internal_timers);

        node_t* current_node = model->get_start_node();
        unsigned int steps = 0;

        while(true)
        {
            if(steps >= options->max_steps_pr_sim)
            {
                break;
            }
            steps++;

            //check current position is valid
            
            // if(!current_node->evaluate_invariants(&lend_internal_timers))
            // {
            //     break;
            // }

            //Progress time
            const double max_progression = current_node->max_time_progression(&lend_internal_timers);
            progress_time(&lend_internal_timers, max_progression, r_state, idx);

            const lend_array<edge_t> outgoing_edges = current_node->get_edges();
            if(outgoing_edges.size() <= 0)
            {
                break;
            }

            edge_t* next_edge = choose_next_edge(&outgoing_edges, r_state, idx);
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

        output[sim_id] = current_node->get_id();
    }

    internal_timers.free_array();
}


void cuda_simulator::simulate(const stochastic_model_t* model, simulation_strategy* strategy)
{
    //setup start variables
    const steady_clock::time_point start = steady_clock::now();
    const int total_simulations = strategy->total_simulations();

    //setup random state
    curandState* state;
    cudaMalloc(&state, sizeof(curandState) * strategy->parallel_degree * strategy->threads_n);

    //setup results array
    int* results = nullptr;
    cudaMalloc(&results, sizeof(int)*total_simulations);


    //!TODO move stochastic model to device memory
    //implement here

    const model_options options = {
        strategy->simulation_amounts,
        strategy->max_sim_steps,
        static_cast<unsigned long>(time(nullptr))
    };

    //run simulations
    simulate_gpu<<<strategy->parallel_degree, strategy->threads_n>>>(
        model, &options, state, results);

    //wait for all processes to finish
    cudaDeviceSynchronize();

    std::cout << "I ran for: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "[ms] \n";

    cudaFree(results);
    cudaFree(state);
}
