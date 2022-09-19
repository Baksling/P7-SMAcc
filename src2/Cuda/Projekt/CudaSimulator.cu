#include "CudaSimulator.h"
#define GPU __device__
#define CPU __host__
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void simulate_d(node* nodes, edge* edges, guard* guards, update* updates, timer* timers, int* result)
{
    
    for (int i = 0; i < 2; i ++)
    {
        printf("%d", nodes[i].get_id());
    }

    *result = 1;
}



cuda_simulator::cuda_simulator(array_info<node>* nodes, array_info<edge>* edges, array_info<guard>* guards, array_info<update>* updates, array_info<timer>* timers)
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
    node* nodes_d;
    edge* edges_d;
    guard* guards_d;
    update* updates_d;
    timer* timers_d;
    int* result_d;

    // Host pointers!
    int result = 0;

    // Allocate memory on device
    cudaMalloc((void**)&nodes_d, sizeof(node) * this->nodes_->size);
    cudaMalloc((void**)&edges_d, sizeof(edge) * this->edges_->size);
    cudaMalloc((void**)&guards_d, sizeof(guard) * this->guards_->size);
    cudaMalloc((void**)&updates_d, sizeof(update) * this->updates_->size);
    cudaMalloc((void**)&timers_d, sizeof(timer) * this->timers_->size);
    cudaMalloc((void**)&result_d, sizeof(int));


    // Copy memory to device!
    cudaMemcpy(nodes_d, this->nodes_->arr, sizeof(node) * this->nodes_->size, cudaMemcpyHostToDevice);
    cudaMemcpy(edges_d, this->edges_->arr, sizeof(edge) * this->edges_->size, cudaMemcpyHostToDevice);
    cudaMemcpy(guards_d, this->guards_->arr, sizeof(guard) * this->guards_->size, cudaMemcpyHostToDevice);
    cudaMemcpy(updates_d, this->updates_->arr, sizeof(update) * this->updates_->size, cudaMemcpyHostToDevice);
    cudaMemcpy(timers_d, this->timers_->arr, sizeof(timer) * this->timers_->size, cudaMemcpyHostToDevice);

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
    node nodes[2] = {node(1), node(2)};
    edge edges[3] = {edge(), edge(), edge()};
    guard guards[1] = {guard(logical_operator::greater_equal, 10, 1)};
    update updates[1] = {update(1, 0)};
    timer timers[1] = {timer(0)};

    array_info<node> n {nodes, 2};
    array_info<edge> e {edges, 3};
    array_info<guard> g {guards, 1};
    array_info<update> u {updates, 1};
    array_info<timer> t {timers, 1};
    
    cuda_simulator sim(&n, &e, &g, &u, &t);
    sim.simulate(10);
    
    return 0;
}


