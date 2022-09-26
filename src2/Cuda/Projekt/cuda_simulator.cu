#include "cuda_simulator.h"

#define GPU __device__
#define CPU __host__
#include "uneven_list.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <list>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <iostream>

#include "stochastic_model.h"

using namespace std::chrono;
using namespace std;


GPU bool validate_guards(const array_info<guard_d>* guards, const array_info<timer_d>* timers)
{
    for (int j = 0; j < guards->size; j++)
    {
        //get timer required by guard.
        const int timer_id = guards->arr[j].get_timer_id();
        //validate guard using required timer.
        if(guards->arr[j].validate(timers->arr[timer_id].get_value())) continue;

        //if validate fails, return false. Also breaks loop
        return false;
    }
    return true;
}

GPU array_info<edge_d> validate_edges(const array_info<edge_d>* edges, const stochastic_model* model, const array_info<timer_d>* timers)
{
    //count of valid edges
    int validated_i = 0;
    
    //buffer of all possible valid edges using validated_i
    const auto valid_edges = static_cast<edge_d*>(malloc(sizeof(edge_d) * edges->size)); 

    //go through all edges from current node.
    for(int i = 0; i < edges->size; i++)
    {
        //find all guards of current edge and validate its guards
        array_info<guard_d> guards = model->get_edge_guards(edges->arr[i].get_id());
        bool validated =  validate_guards(&guards, timers);
        guards.free_arr();

        //only continue if all guards are valid.
        if(!validated) continue;

        //check all guards of destination node.
        guards = model->get_node_invariants(edges->arr[i].get_dest_node());
        validated =  validate_guards(&guards, timers);
        guards.free_arr();

        //add to valid_edges if both check succeed.
        if (validated)
        {
            valid_edges[validated_i] = edges->arr[i];
            validated_i++;
        }
    }

    //copy all valid edges into appropriate sized array.
    const auto result_arr = static_cast<edge_d*>(malloc(sizeof(edge_d) * validated_i));
    for (int i = 0; i < validated_i; i++)
    {
        result_arr[i] = valid_edges[i];
    }
    //free old buffer
    free(valid_edges);

    const array_info<edge_d> result { result_arr, validated_i};
    return result;
}

GPU edge_d* choose_next_edge(const array_info<edge_d>* edges, curandState* states, const unsigned int thread_id)
{
    //if no possible edges, return null pointer
    if(edges->size == 0) return nullptr;

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const float r = 1.0f - curand_uniform(&states[thread_id]); 
    const int index = static_cast<int>(r * static_cast<float>(edges->size));

    return &edges->arr[index] ;
}

GPU void progress_time(const array_info<timer_d>* timers, const double difference, curandState* states, const unsigned int thread_id)
{
    //Get random uniform value between ]0.0f, 0.1f] * difference gives a random uniform range of ]0, diff]
    const double time_progression = difference * curand_uniform_double(&states[thread_id]);

    //update all timers by adding time_progression to each
    for(int i = 0; i < timers->size; i++)
    {
        timers->arr[i].add_time(time_progression);
    }
}

//Finds the furthest possible time it is possible to progress in current step.
GPU double find_least_difference(const array_info<guard_d>* invariants, const array_info<timer_d>* timers,
    const int max_value = 100)
{
    double least_difference = max_value;
    

    //check all guards of current node
    for (int i = 0; i < invariants->size; i++)
    {
        const logical_operator guard_type = invariants->arr[i].get_type();
        //only relevant if it is upper bounded logical operator.
        if(guard_type != logical_operator::less_equal && guard_type != logical_operator::less) continue;

        //find difference in upper bounded guard value and current time.
        const double diff = invariants->arr[i].get_value() - timers->arr[invariants->arr[i].get_timer_id()].get_value();
        //if equal or higher than 0 and its smallest value, find newest lower bound.
        if (diff >= 0 && diff < least_difference)
            least_difference = diff;
    }

    //free index.
    return least_difference;
}

GPU void reset_timers(const array_info<timer_d>* timers, const array_info<timer_d>* original_time)
{
    
}


GPU void free_all(const array_info<guard_d>* arr1, const array_info<edge_d>* arr2, const array_info<edge_d>* arr3)
{
    if(arr1 != nullptr) free(arr1->arr);
    if(arr2 != nullptr) free(arr2->arr);
    if(arr3 != nullptr) free(arr3->arr);
}


__global__ void simulate_d_2(
    stochastic_model* model,
    const unsigned long seed,
    curandState* r_state,
    int* output
    )
{
    //init variables and random state
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, idx, &r_state[idx]);
    constexpr int max_number_of_steps = 1000;
    output[idx] = 0;


    // init local timers.
    constexpr int repeat_sim = 1000;
    const array_info<timer_d> internal_timers = model->copy_timers();


    for (int test = 0; test < repeat_sim; test++)
    {
        model->reset_timers(&internal_timers);
        
        int current_node = 0;
        int steps = 0;

        while (true)
        {
            if(steps >= max_number_of_steps)
            {
                break;
            }
            steps++;

            const array_info<guard_d> invariants = model->get_node_invariants(current_node);
            if (!validate_guards(&invariants, &internal_timers))
            {
                printf("2");
                invariants.free_arr();
                break;
            }

            
            const double difference = find_least_difference(&invariants, &internal_timers);
            progress_time(&internal_timers, difference, r_state, idx);
            invariants.free_arr();
            
            const array_info<edge_d> edges = model->get_node_edges(current_node);
            if (edges.size <= 0)
            {
                edges.free_arr();
                continue;
            }

            const array_info<edge_d> valid_edges = validate_edges(&edges, model, &internal_timers);
            edge_d* edge = choose_next_edge(&valid_edges, r_state, idx);

            
            if(edge == nullptr)
            {
                //printf("Stopped at node: %d \n", current_node);
                edges.free_arr();
                valid_edges.free_arr();
                continue;
            }
            
            printf("going from: %d to %d \n", current_node, edge->get_dest_node());
            current_node = edge->get_dest_node();
            edges.free_arr();
            valid_edges.free_arr();

            if(model->is_goal_node(current_node))
            {
                output[idx]++;
                break;
            }
        }
    }

    internal_timers.free_arr();
    const float proc = (static_cast<float>(output[idx])/static_cast<float>(repeat_sim))*100;
    printf("Hit goal state: %d / %d times (%f)%% \n", output[idx], repeat_sim, proc);
}

cuda_simulator::cuda_simulator()
{
}

void copy_to_device(void* dest, const void* src, const int size)
{
    cudaMalloc(((void**)&dest), size);
    cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

void cuda_simulator::simulate_2(uneven_list<edge_d> *node_to_edge, uneven_list<guard_d> *node_to_invariant,
    uneven_list<guard_d> *edge_to_guard, uneven_list<update_d> *edge_to_update, int timer_amount, timer_d *timers) const
{
    const steady_clock::time_point start = steady_clock::now();

    constexpr int parallel_degree = 1;
    constexpr int threads_n = 1;
    
    curandState* state;
    cudaMalloc(&state, sizeof(curandState)*parallel_degree*threads_n);
    int* results;
    cudaMalloc(&results, sizeof(int)*parallel_degree*threads_n);

    time_t t;
    t = time(&t);
    
    stochastic_model* model_d = nullptr;
    stochastic_model model(node_to_edge, node_to_invariant,
        edge_to_guard, edge_to_update, timers, timer_amount);
    cudaMalloc(((void**)&model_d), sizeof(stochastic_model));
    cudaMemcpy(model_d, &model, sizeof(stochastic_model), cudaMemcpyHostToDevice);

    simulate_d_2<<<parallel_degree, threads_n>>>(model_d, static_cast<unsigned long>(t), state, results);

    cudaDeviceSynchronize();

    cout << "I ran for: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "[ms] \n";
    cudaFree(model_d);
    cudaFree(state);
    cudaFree(results);
}

