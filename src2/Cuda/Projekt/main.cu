#include "main.h"
#define GPU __device__
#define CPU __host__
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void simulate_d(node_d* nodes, edge_d* edges, guard_d* guards, update_d* updates, timer_d* timers, int* result)
{
    
    for (int i = 0; i < 2; i ++)
    {
        printf("%d", nodes[i].get_id());
    }

    *result = 1;
}



cuda_simulator::cuda_simulator(array_info<node_d>* nodes, array_info<edge_d>* edges, array_info<guard_d>* guards, array_info<update_d>* updates, array_info<timer_d>* timers)
{
    this->nodes_ = nodes;
    this->edges_ = edges;
    this->guards_ = guards;
    this->updates_ = updates;
    this->timers_ = timers;
}

__host__ void cuda_simulator::simulate(int max_nr_of_steps)
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
}

int main()
{
    node_d nodes[2] = {node_d(1), node_d(2)};
    edge_d edges[2] = {edge_d(1, 2), edge_d(2, 1)};
    guard_d guards[1] = {guard_d(1, 1, logical_operator::greater_equal, 10)};
    update_d updates[1] = {update_d(1, 1, 0)};
    timer_d timers[1] = {timer_d(1,0)};

    array_info<node_d> n {nodes, 2};
    array_info<edge_d> e {edges, 2};
    array_info<guard_d> g {guards, 1};
    array_info<update_d> u {updates, 1};
    array_info<timer_d> t {timers, 1};
    
    cuda_simulator sim(&n, &e, &g, &u, &t);
    sim.simulate(10);
    
    return 0;
}


