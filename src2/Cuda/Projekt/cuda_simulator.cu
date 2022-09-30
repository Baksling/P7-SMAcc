#include "cuda_simulator.h"

#define GPU __device__
#define CPU __host__
#define NOT_GOAL_STATE -1
#include "uneven_list.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <unordered_map>

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

    //summed weight
    float weight_sum = 0.0f;
    for(int i = 0; i < edges->size; i++)
        weight_sum += edges->arr[i].get_weight();

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const float r_val = (1.0f - curand_uniform(&states[thread_id]))*weight_sum;
    float r_acc = 0.0; 

    //pick the weighted random value.
    for (int i = 0; i < edges->size; ++i)
    {
        r_acc += edges->arr[i].get_weight();
        if(r_val < r_acc) return &edges->arr[i];
    }

    //This should be handled in for loop.
    //This is for safety :)
    return &edges->arr[edges->size - 1];
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

struct model_options
{
    int simulation_amount;
    int max_steps_pr_sim;
    unsigned long seed;
};

__global__ void simulate_d_2(
    const stochastic_model* model,
    const model_options* options,
    curandState* r_state,
    int* output
    )
{
    // printf("HELOL\n");
    // model->pretty_print();
    // printf("HELOLo");
    //init variables and random state
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(options->seed, idx, idx, &r_state[idx]);

    // init local timers.
    const array_info<timer_d> internal_timers = model->copy_timers();

    for (int i = 0; i < options->simulation_amount; i++)
    {
        //reset current location
        const int sim_id = i + options->simulation_amount * static_cast<int>(idx);
        output[sim_id] = NOT_GOAL_STATE;
        
        //reset timers through each simulation
        model->reset_timers(&internal_timers);
        
        int current_node = model->get_start_node();
        int steps = 0;

        while (true)
        {
            if(steps >= options->max_steps_pr_sim)
            {
                break;
            }
            steps++;

            const array_info<guard_d> invariants = model->get_node_invariants(current_node);
            if (!validate_guards(&invariants, &internal_timers))
            {
                invariants.free_arr();
                break;
            }

            if (!model->is_branchpoint(current_node))
            {
                const double difference = find_least_difference(&invariants, &internal_timers);
                progress_time(&internal_timers, difference, r_state, idx);
            }
            
            invariants.free_arr();
            
            const array_info<edge_d> edges = model->get_node_edges(current_node);
            if (edges.size <= 0) //current edge has no outgoing edges.
            {
                edges.free_arr();
                break;
            }

            const array_info<edge_d> valid_edges = validate_edges(&edges, model, &internal_timers);
            edge_d* edge = choose_next_edge(&valid_edges, r_state, idx);
            
            if(edge == nullptr) //no traversal can be done in current step
            {
                // printf("Stopped at node: %d \n", current_node);
                edges.free_arr();
                valid_edges.free_arr();
                continue;
            }

            model->traverse_edge_update(edge->get_id(), &internal_timers);
            current_node = edge->get_dest_node();
            edges.free_arr();
            valid_edges.free_arr();

            if(model->is_goal_node(current_node))
            {
                break;
            }
        }

        //record final location
        output[sim_id] = current_node;
    }

    internal_timers.free_arr();
}

cuda_simulator::cuda_simulator() = default;

void copy_to_device(void* dest, const void* src, const int size)
{
    cudaMalloc(((void**)&dest), size);
    cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

float calc_percentage(const int counter, const int divisor)
{
    return (static_cast<float>(counter)/static_cast<float>(divisor))*100;
} 

void print_results(unordered_map<int,int>* result_map, const int result_size)
{
    for (auto& it : (*result_map))
    {
        if(it.first == NOT_GOAL_STATE) continue;
        const float percentage = calc_percentage(it.second, result_size);
        cout << "Node: " << it.first << " reached " << it.second << " times. (" << percentage << ")%\n";
    }
    const float percentage = calc_percentage((*result_map)[NOT_GOAL_STATE], result_size);
    cout << "No goal state was reached " << (*result_map)[NOT_GOAL_STATE] << " times. (" << percentage << ")%\n";
    cout << "Nr of simulations: " << result_size << "\n";
}

void count_results(const int total_simulations,
    const int* local_results,
    unordered_map<int, int>* node_results)
{
    node_results->insert_or_assign(NOT_GOAL_STATE, 0);
    for (int i = 0; i < total_simulations; i++)
    {
        const int key = local_results[i];
        const int value = node_results->count(key) == 1
                              ? (*node_results)[key]+1
                              : 1;
        node_results->insert_or_assign(key, value);
    }
}


//Expects all parameters to be on the host.
void cuda_simulator::simulate(const stochastic_model* model, const simulation_strategy* strategy, list<void*>* free_list) const
{
    const steady_clock::time_point start = steady_clock::now();

    const int total_simulations = strategy->total_simulations();
    
    curandState* state;
    cudaMalloc(&state, sizeof(curandState) * strategy->parallel_degree * strategy->threads_n);
    
    int* results = nullptr;
    cudaMalloc(&results, sizeof(int)*total_simulations);
    
    //move model to device
    
    stochastic_model* model_d = nullptr;
    model->cuda_allocate(&model_d, free_list);
    //move options to device
    model_options* options_d = nullptr;
    const model_options options = {
        strategy->simulation_amounts,
        strategy->max_sim_steps,
        static_cast<unsigned long>(time(nullptr))
    };
    cudaMalloc(&options_d, sizeof(model_options));
    cudaMemcpy(options_d, &options, sizeof(model_options), cudaMemcpyHostToDevice);
    
    //run simulations
    simulate_d_2<<<strategy->parallel_degree, strategy->threads_n>>>(
        model_d, options_d, state, results);

    //wait for all processes to finish
    cudaDeviceSynchronize();

    //report time spent simulating
    cout << "I ran for: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "[ms] \n";

    //copy results from device to host
    int* local_results = static_cast<int*>(malloc(sizeof(int)*total_simulations));
    cudaMemcpy(local_results, results, sizeof(int)*total_simulations, cudaMemcpyDeviceToHost);

    //count the results
    unordered_map<int, int> node_results;
    count_results(total_simulations, local_results, &node_results);

    print_results(&node_results, total_simulations);

    //free heap allocated and cuda memory
    //model is not handled here, as it is added to the free list.
    //only the local instantiated memory is freed here.
    free(local_results);
    cudaFree(results);
    cudaFree(options_d);
    cudaFree(state);
}




