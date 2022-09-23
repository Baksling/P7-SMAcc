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

using namespace std::chrono;
using namespace std;


GPU array_info<edge_d> validate_edges(const array_info<edge_d>* edges, uneven_list<guard_d>* guard_ulist,
                                      uneven_list<guard_d>* guard_node_ulist, array_info<timer_d>* timers)
{
    int validated_i = 0;
    const auto valid_edges = static_cast<edge_d*>(malloc(sizeof(edge_d) * edges->size));

    for(int i = 0; i < edges->size; i++)
    {
        bool validated = true;
        const array_info<guard_d> guards = guard_ulist->get_index(edges->arr[i].get_id());

        for (int j = 0; j < guards.size; j++)
        {
            const int timer_id = guards.arr[j].get_timer_id();
            if(guards.arr[j].validate(timers->arr[timer_id].get_value())) continue;

            validated = false;
            break;
        }
        free(guards.arr);
        if(!validated) continue;

        const array_info<guard_d> guards_node = guard_node_ulist->get_index(edges->arr[i].get_dest_node());
        for (int j = 0; j < guards_node.size; j++)
        {
            const int timer_id = guards_node.arr[j].get_timer_id();
            if(guards_node.arr[j].validate(timers->arr[timer_id].get_value())) continue;

            validated = false;
            break;
        }
        free(guards_node.arr);

        if (validated)
        {
            valid_edges[validated_i] = edges->arr[i];
            validated_i++;
        }
    }

    edge_d* result_arr = (edge_d*)malloc(sizeof(edge_d) * validated_i);
    for (int i = 0; i < validated_i; i++)
    {
        result_arr[i] = valid_edges[i];
    }

    free(valid_edges);

    const array_info<edge_d> result { result_arr, validated_i};
    return result;
}

GPU edge_d* choose_next_edge(array_info<edge_d>* edges, curandState* states, const unsigned int thread_id)
{

    if(edges->size == 0) return nullptr;

    const float r = 1.0f - curand_uniform(&states[thread_id]);
    const int index = (int)(r * (float)edges->size);

    // printf("MY RANDOM IS: %f | MY INDEX IS: %d | size: %d | blockIdx: %d\n", r, index, edges->size, blockIdx.x);
    
    // printf("index: %d .. r: %f .. size: %d %f % f\n", index, r, edges.size, (float) edges.size, r * (float)edges.size);
    // printf("edge: %d Moving from %d to %d \n", edges.arr[index].get_id(), current_node, edges.arr[index].get_dest_node());

    return &edges->arr[index] ;
    
}

GPU void progress_time(const array_info<timer_d>* timers, const double difference, curandState* states, const unsigned int thread_id)
{
    const double time_progression = difference * curand_uniform_double(&states[thread_id]);
    
    for(int i = 0; i < timers->size; i++)
    {
        timers->arr[i].set_value(timers->arr[i].get_value() + time_progression);
    }
}

GPU double find_least_difference(int current_node, uneven_list<guard_d>* node_to_invariant, array_info<timer_d>* timers, const int max_value = 100)
{
    double least_difference = max_value;

    const array_info<guard_d> guards = node_to_invariant->get_index(current_node);

    for (int i = 0; i < guards.size; i++)
    {
        const logical_operator guard_type = guards.arr[i].get_type();
        if(guard_type != logical_operator::less_equal && guard_type != logical_operator::less) continue;

        const double diff = guards.arr[i].get_value() - timers->arr[guards.arr[i].get_timer_id()].get_value();
        if (diff >= 0 && diff < least_difference)
            least_difference = diff;
    }

    free(guards.arr);

    return least_difference;
}

__global__ void simulate_d(node_d* nodes, edge_d* edges, guard_d* guards, update_d* updates, timer_d* timers, int* result)
{
    
    
    for (int i = 0; i < 2; i ++)
    {
        printf("%d", nodes[i].get_id());
    }

    *result = 1;
}

GPU void free_all(array_info<guard_d>* arr1, array_info<edge_d>* arr2, array_info<edge_d>* arr3)
{
    if(arr1 != nullptr) free(arr1->arr);
    if(arr2 != nullptr) free(arr2->arr);
    if(arr3 != nullptr) free(arr3->arr);
}

__global__ void simulate_d_2(
    uneven_list<edge_d>* node_to_edge,
    uneven_list<guard_d>* node_to_invariant,
    uneven_list<guard_d>* edge_to_guard,
    uneven_list<update_d>* egde_to_update,
    timer_d* timers, int timer_amount,
    curandState* states,
    int* output, unsigned long seed
    )
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // printf("MY_SEED: %d | threadIdx: %d | blockDim: %d | blockIdx: %d\n", idx, threadIdx.x, blockDim.x, blockIdx.x);
    curand_init(seed, idx, idx, &states[idx]);
    const int max_number_of_steps = 1000;
    output[idx] = 0;
    // int times_hit_maximum = 0;

    const auto timer_copies = static_cast<timer_d*>(malloc(sizeof(timer_d) * timer_amount));

    for (int i = 0; i < timer_amount; i++)
    {
        timer_copies[i] = timers[i].copy();
    }
    array_info<timer_d> internal_timers{ timer_copies, timer_amount};

    for (int test = 0; test < 1100; test++)
    {
        // if(test % 20 == 0) printf("%d \n", test);

        //Reset timers!
        for (int i = 0; i < timer_amount; i++)
        {
            internal_timers.arr[i].set_value(timers[i].get_value());
        }
        
        // printf("Starting new run --- \n");
        int current_node = 0;
        // int last_node = -1;
        int steps = 0;

        while (true)
        {
            array_info<guard_d> invariants{ nullptr, 0};
            array_info<edge_d> edges{ nullptr, 0 };
            array_info<edge_d> valid_edges{ nullptr, 0 };
            
            if(steps >= max_number_of_steps)
            {
                // printf("Hit max steps!");
                output[idx]++;
                break;
            }
            steps++;

            invariants = node_to_invariant->get_index(current_node);
            bool valid_node = true;
            // printf("ARGHH %d \n", invariants.size);
            for(int i = 0; i < invariants.size; i++)
            {
                const int timer_id = invariants.arr[i].get_timer_id();
                if(invariants.arr[i].validate(internal_timers.arr[timer_id].get_value())) continue;


                // printf("WHAT HAPPEND?!");
                output[idx]++;
                valid_node = false;
            }

            if (!valid_node)
            {
                free_all(&invariants, nullptr, nullptr);
                break;
            }

            const double difference = find_least_difference(current_node, node_to_invariant, &internal_timers);
            // printf("TIME MOVES DIFFERENTLY! %f \n", difference);
            progress_time(&internal_timers, difference, states, idx);
            
            edges = node_to_edge->get_index(current_node);
            if (edges.size <= 0)
            {
                free_all(&invariants, &edges, nullptr);
                continue;
            }

            valid_edges = validate_edges(&edges, edge_to_guard, node_to_invariant, &internal_timers);
            edge_d* edge = choose_next_edge(&valid_edges, states, idx);

            if(edge == nullptr)
            {
                // printf("Stopped at node: %d \n", current_node);
                free_all(&invariants, &edges, &valid_edges);
                continue;
            }
            
            current_node = edge->get_dest_node();

            if (current_node == 2)
            {
                // printf("WIN!");
                free_all(&invariants, &edges, &valid_edges);
                break;
            }

            // if(last_node == current_node)
            // {
            //     // printf("Stopped at node: %d \n", current_node);
            //     // break;
            // }

            // last_node = current_node;
            free_all(&invariants, &edges, &valid_edges);
        
        }
    }

    free(timer_copies);

    //cout << "1000 runs!: " <<  << "[ns] \n";
    printf("Hit maximum steps: %d times \n", output[idx]);
    
}

cuda_simulator::cuda_simulator(
    array_info<node_d>* nodes, array_info<edge_d>* edges,
    array_info<guard_d>* guards, array_info<update_d>* updates,
    array_info<timer_d>* timers)
{
    this->nodes_ = nodes;
    this->edges_ = edges;
    this->guards_ = guards;
    this->updates_ = updates;
    this->timers_ = timers;
}

CPU void cuda_simulator::simulate(int max_nr_of_steps)
{
    // Device pointers
    node_d* nodes_d;
    edge_d* edges_d;
    guard_d* guards_d;
    update_d* updates_d;
    timer_d* timers_d;
    int* result_d;

    // Host pointers!
    int result = 0;

    // Allocate memory on device
    cudaMalloc((void**)&nodes_d, sizeof(node_d) * this->nodes_->size);
    cudaMalloc((void**)&edges_d, sizeof(edge_d) * this->edges_->size);
    cudaMalloc((void**)&guards_d, sizeof(guard_d) * this->guards_->size);
    cudaMalloc((void**)&updates_d, sizeof(update_d) * this->updates_->size);
    cudaMalloc((void**)&timers_d, sizeof(timer_d) * this->timers_->size);
    cudaMalloc((void**)&result_d, sizeof(int));
    
    // Copy memory to device!
    cudaMemcpy(nodes_d, this->nodes_->arr, sizeof(node_d) * this->nodes_->size, cudaMemcpyHostToDevice);
    cudaMemcpy(edges_d, this->edges_->arr, sizeof(edge_d) * this->edges_->size, cudaMemcpyHostToDevice);
    cudaMemcpy(guards_d, this->guards_->arr, sizeof(guard_d) * this->guards_->size, cudaMemcpyHostToDevice);
    cudaMemcpy(updates_d, this->updates_->arr, sizeof(update_d) * this->updates_->size, cudaMemcpyHostToDevice);
    cudaMemcpy(timers_d, this->timers_->arr, sizeof(timer_d) * this->timers_->size, cudaMemcpyHostToDevice);

    //Run program
    simulate_d<<<1,1>>>(nodes_d, edges_d, guards_d, updates_d, timers_d, result_d);
    
    // Copy result to
    cudaMemcpy(&result, result_d, sizeof(int), cudaMemcpyDeviceToHost);

    //printf("%d", result);

    //Free device memory
    cudaFree(nodes_d);
    cudaFree(edges_d);
    cudaFree(guards_d);
    cudaFree(updates_d);
    cudaFree(timers_d);
    cudaFree(result_d);
}

void cuda_simulator::simulate_2(uneven_list<edge_d> *node_to_edge, uneven_list<guard_d> *node_to_invariant,
    uneven_list<guard_d> *edge_to_guard, uneven_list<update_d> *edge_to_update, int timer_amount, timer_d *timers) const
{
    const steady_clock::time_point start = steady_clock::now();

    constexpr int parallel_degree = 32;
    constexpr int threads_n = 512;
    
    curandState* state;
    cudaMalloc((void**)&state, sizeof(curandState)*parallel_degree*threads_n);
    int* results;
    cudaMalloc((void**)&results, sizeof(int)*parallel_degree*threads_n);

    time_t t;
    time(&t);
    simulate_d_2<<<parallel_degree, threads_n>>>(node_to_edge, node_to_invariant,
        edge_to_guard, edge_to_update, timers, timer_amount, state, results, static_cast<unsigned long>(t));

    cudaDeviceSynchronize();

    cout << "I ran for: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "[ms] \n";
    cudaFree(state);
    cudaFree(results);
}

cuda_simulator::cuda_simulator()
{
    
}

