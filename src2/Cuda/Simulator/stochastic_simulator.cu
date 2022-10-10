
#ifndef STOCHASTIC_SIMULATOR_CU
#define STOCHASTIC_SIMULATOR_CU

#include "../Domain/common.h"
#include "simulation_strategy.h"
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


CPU GPU void init_curand_states(const unsigned long long seed,
                                const unsigned long long subsequence,
                                const unsigned long long offset,
                                curandStateXORWOW_t *state)
{
    // Break up seed, apply salt
    // Constants are arbitrary nonzero values
    const unsigned int s0 = static_cast<unsigned int>(seed) ^ 0xaad26b49UL;
    const unsigned int s1 = static_cast<unsigned int>(seed >> 32) ^ 0xf7dcefddUL;
    // Simple multiplication to mix up bits
    // Constants are arbitrary odd values
    const unsigned int t0 = 1099087573UL * s0;
    const unsigned int t1 = 2591861531UL * s1;
    state->d = 6615241 + t1 + t0;
    state->v[0] = 123456789UL + t0;
    state->v[1] = 362436069UL ^ t0;
    state->v[2] = 521288629UL + t1;
    state->v[3] = 88675123UL ^ t1;
    state->v[4] = 5783321UL + t0;
    _skipahead_sequence_inplace<curandStateXORWOW_t, 5>(subsequence, state);
    _skipahead_inplace<curandStateXORWOW_t, 5>(offset, state);
    state->boxmuller_flag = 0;
    state->boxmuller_flag_double = 0;
    state->boxmuller_extra = 0.f;
    state->boxmuller_extra_double = 0.;
}

CPU GPU unsigned int curand_state(curandStateXORWOW_t *state)
{
    unsigned int t;
    t = (state->v[0] ^ (state->v[0] >> 2));
    state->v[0] = state->v[1];
    state->v[1] = state->v[2];
    state->v[2] = state->v[3];
    state->v[3] = state->v[4];
    state->v[4] = (state->v[4] ^ (state->v[4] <<4)) ^ (t ^ (t << 1));
    state->d += 362437;
    return state->v[4] + state->d;
}

CPU GPU double generate_random_double(curandState* state)
{
    const unsigned int x = curand_state(state);
    const unsigned int y = curand_state(state);
    const unsigned long long z = static_cast<unsigned long long>(x) ^
        (static_cast<unsigned long long>(y) << (53 - 32));
    
    return static_cast<double>(z) * (1.1102230246251565e-16) + ((1.1102230246251565e-16)/2.0);
}

CPU GPU float generate_random_float(curandState* state)
{
    return static_cast<float>(curand_state(state)) * (2.3283064e-10f) + ((2.3283064e-10f)/2.0f);
}

CPU GPU edge_t* choose_next_edge(const lend_array<edge_t*>* edges, const lend_array<clock_timer_t>* timer_arr,
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
    if(valid_edges == nullptr) printf("COULD NOT ALLOCATE HEAP MEMORY\n");
    int valid_count = 0;
    
    for (int i = 0; i < edges->size(); ++i)
    {
        valid_edges[i] = nullptr; //clean malloc
        edge_t* edge = edges->get(i);
        if(edge->evaluate_constraints(timer_arr))
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
    const float r_val = (1.0f - generate_random_float(&states[thread_id]))*weight_sum;
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

CPU GPU void progress_time(const lend_array<clock_timer_t>* timers, const double difference, curandState* states, const unsigned int thread_id)
{
    //Get random uniform value between ]0.0f, 0.1f] * difference gives a random uniform range of ]0, diff]
    const double time_progression = difference * generate_random_double(&states[thread_id]);

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
    const unsigned long idx
)
{
    init_curand_states(options->seed, idx, idx, &r_state[idx]);
    // curand_init(options->seed, idx, idx, &r_state[idx]);

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
                const double max_progression = current_node->max_time_progression(&lend_internal_timers);
                progress_time(&lend_internal_timers, max_progression, r_state, idx);
            }

            const lend_array<edge_t*> outgoing_edges = current_node->get_edges();
            if(outgoing_edges.size() <= 0)
            {
                hit_max_steps = false;
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
            output[sim_id] = current_node->get_id();
        }
    }

    internal_timers.free_array();
}


__global__ void simulate_gpu(
    const stochastic_model_t* model,
    const model_options* options,
    curandState* r_state,
    int* output
)
{
    const unsigned long idx = threadIdx.x + blockDim.x * blockIdx.x;
    simulate_stochastic_model(model, options, r_state, output, idx);
}

void read_results(const int* cuda_results, const unsigned long total_simulations, std::map<int, unsigned long>* results)
{
    int* local_results = static_cast<int*>(malloc(sizeof(int)*total_simulations));
    cudaMemcpy(local_results, cuda_results, sizeof(int)*total_simulations, cudaMemcpyDeviceToHost);

    for (unsigned long i = 0; i < total_simulations; ++i)
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

#endif
