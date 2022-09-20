#include "cuda_simulator.h"

#define GPU __device__
#define CPU __host__
#include "uneven_list.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <list>
#include <stdio.h>
#include <math.h>


GPU edge_d* choose_next_edge(array_info<edge_d> edges, curandState* states)
{

    if(edges.size == 0) return nullptr;
    
    float r = 1.0f - curand_uniform(&states[0]);
    int index = (int)(r * (float)edges.size);

    printf("index: %d .. r: %f .. size: %d %f % f\n", index, r, edges.size, (float) edges.size, r * (float)edges.size);
    printf("Moving from %d to %d \n", edges.arr[index].get_id(), edges.arr[index].get_dest_node());

    return &edges.arr[index] ;
    
}

__global__ void simulate_d(node_d* nodes, edge_d* edges, guard_d* guards, update_d* updates, timer_d* timers, int* result)
{
    
    
    for (int i = 0; i < 2; i ++)
    {
        printf("%d", nodes[i].get_id());
    }

    *result = 1;
}

__global__ void simulate_d_2(
    uneven_list<edge_d>* node_to_edge,
    uneven_list<guard_d>* node_to_invariant,
    uneven_list<guard_d>* edge_to_guard,
    uneven_list<update_d>* egde_to_update,
    timer_d* timers, curandState* states
    )
{

    unsigned int id = threadIdx.x + blockDim.x;
    int seed = id;
    curand_init(seed, id, 0, &states[0]);
    
    
    for (int test = 0; test < 1000; test++)
    {
        printf("Starting new run --- \n");
        int current_node = 0;
        int last_node = -1;

        while (true)
        {

            array_info<edge_d> edges = node_to_edge->get_index(current_node);
            edge_d* edge = choose_next_edge(edges, states);

            if(edge == nullptr)
            {
                printf("Stopped at node: %d \n", current_node);
                break;
            }
            
            current_node = edge->get_dest_node();

            if(last_node == current_node)
            {
                printf("Stopped at node: %d \n", current_node);
                break;
            }

            last_node = current_node;
        
        }
    }
    
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
    uneven_list<guard_d> *edge_to_guard, uneven_list<update_d> *edge_to_update, timer_d *timers)
{
    curandState* state;
    cudaMalloc((void**)&state, sizeof(curandState));

    
    simulate_d_2<<<1,1>>>(node_to_edge, node_to_invariant, edge_to_guard, edge_to_update, timers, state);
}

cuda_simulator::cuda_simulator()
{
    
}

