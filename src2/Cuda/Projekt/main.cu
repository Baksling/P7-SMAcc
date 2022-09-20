#include "main.h"
#include "common.h"
#include "uneven_list.h"
#include <list>
#include <stdio.h>

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
}

int main()
{
    node_d nodes[2] = {node_d(1), node_d(2)};
    edge_d edges[2] = {edge_d(1, 2), edge_d(2, 1)};
    guard_d guards[1] = {guard_d(1, logical_operator::greater_equal, 10)};
    update_d updates[1] = {update_d(1, 0)};
    timer_d timers[1] = {timer_d(1,0)};

    array_info<node_d> n {nodes, 2};
    array_info<edge_d> e {edges, 2};
    array_info<guard_d> g {guards, 1};
    array_info<update_d> u {updates, 1};
    array_info<timer_d> t {timers, 1};
    
    cuda_simulator sim(&n, &e, &g, &u, &t);
    // sim.simulate(10);

    // node = [node]
    // edge = [[edge]], hvor index af første = node index
    // invarient = [[guard]], hvor index af først = node index
    // guard = [guard], hvor index = edge id
    // update = [update], hvor index = edge id
    // timer = [timer], index = id.

    // -----------------------------------------------------------

    //Nodes
    list<node_d> nodes__;
    nodes__.emplace_back(0);
    nodes__.emplace_back(1);
    nodes__.emplace_back(2);

    // Edges
    list<edge_d> edges_1_;
    edges_1_.emplace_back(0, 1);
    edges_1_.emplace_back(1,2);

    list<edge_d> edges_2_;
    edges_2_.emplace_back(2, 1);

    list<edge_d> edges_3_;

    list<list<edge_d>> edge_list;
    edge_list.push_back(edges_1_);
    edge_list.push_back(edges_2_);
    edge_list.push_back(edges_3_);

    //Invariants for nodes
    list<guard_d> invariant_1_;
    invariant_1_.emplace_back(0, logical_operator::less, 10);

    list<guard_d> invariant_2_;
    invariant_2_.emplace_back(0, logical_operator::less_equal, 10);

    list<guard_d> invariant_3_;
    invariant_3_.emplace_back(0, logical_operator::greater_equal, 10);

    list<list<guard_d>> invariant_list;
    invariant_list.push_back(invariant_1_);
    invariant_list.push_back(invariant_2_);
    invariant_list.push_back(invariant_3_);

    // Guard List for edges
    list<guard_d> guard_1_;
    guard_1_.emplace_back(0, logical_operator::less, 10);

    list<guard_d> guard_2_;
    guard_2_.emplace_back(0, logical_operator::less_equal, 10);

    list<guard_d> guard_3_;
    guard_3_.emplace_back(0, logical_operator::greater_equal, 10);

    list<list<guard_d>> guard_list;
    guard_list.push_back(invariant_1_);
    guard_list.push_back(invariant_2_);
    guard_list.push_back(invariant_3_);

    //Update list for edges
    list<update_d> update_1_;
    update_1_.emplace_back(0, 0);
    update_1_.emplace_back(1,0);
    
    list<update_d> update_2_;
    update_2_.emplace_back(0, 0);

    list<update_d> update_3_;

    list<list<update_d>> update_list;
    update_list.push_back(update_1_);
    update_list.push_back(update_2_);
    update_list.push_back(update_3_);

    // Timers
    timer_d* timer_list;
    timer_list = (timer_d*)malloc(sizeof(timer_d) * 2);
    timer_list[0] = timer_d(0, 0);
    timer_list[1] = timer_d(1, 0);
    
    uneven_list<edge_d> node_to_edge(&edge_list, 3);
    uneven_list<guard_d> node_to_invariant(&invariant_list, 3);
    uneven_list<guard_d> edge_to_guard(&guard_list, 3);
    uneven_list<update_d> edge_to_update(&update_list, 3);
    

    array_info<guard_d> hej = node_to_invariant.get_index(0);
    
    for(int i = 0; i < hej.size; i++) {
        printf("%d -> %d", hej.arr[i].get_timer_id(), (int)hej.arr[i].get_value());
    }

    return 0;
}


