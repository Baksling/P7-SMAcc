#include "CudaSimulator.h"
#define GPU __device__
#define CPU __host__
#include <cuda.h>
#include <cuda_runtime.h>

GPU void simulate(node* nodes, edge* edges, guard* guards, update* updates, timer* timers, int* result)
{


    *result = 1;
}



cuda_simulator::cuda_simulator(array_info nodes, array_info edges, array_info guards, array_info updates, array_info timers)
{
    this->nodes_ = (node*)malloc(sizeof(node) * nodes.size);
    this->edges_ = (edge*)malloc(sizeof(edge) * edges.size);
    this->guards_ = (guard*)malloc(sizeof(guard) * guards.size);
    this->updates_ = (update*)malloc(sizeof(update) * updates.size);
    this->timers_ = (timer*)malloc(sizeof(timer) * timers.size);
}

void cuda_simulator::simulate(int max_nr_of_steps)
{
    
    node* nodes = cudaMalloc()
    
}

