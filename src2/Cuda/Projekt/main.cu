#include "main.h"
#include "cuda_map.h"
#define GPU __device__
#define CPU __host__
#include <cuda.h>
#include <cuda_runtime.h>
#include <list>
#include <stdio.h>

#include "uneven_list.h"
#include "node_d.h"
#include "edge_d.h"
#include "guard_d.h"
#include "update_d.h"
#include "timer_d.h"
#include "cuda_simulator.h"
#include "stochastic_model.h"

int main()
{
    //Nodes
    list<node_d> nodes_;
    nodes_.emplace_back(0);
    nodes_.emplace_back(1);
    nodes_.emplace_back(2);
    nodes_.emplace_back(3);
    nodes_.emplace_back(4);

    // Edges
    list<edge_d> edges_0_;
    edges_0_.emplace_back(0, 1, 1);
    edges_0_.emplace_back(1,2, 2);
    edges_0_.emplace_back(2,0, 1);

    list<edge_d> edges_1_;
    edges_1_.emplace_back(3, 3, 2);
    edges_1_.emplace_back(4, 4, 1);

    list<edge_d> edges_2_;
    //edges_2_.emplace_back(3,0);

    list<edge_d> edges_3_;

    list<edge_d> edges_4_;

    list<list<edge_d>> edge_list;
    edge_list.push_back(edges_0_);
    edge_list.push_back(edges_1_);
    edge_list.push_back(edges_2_);
    edge_list.push_back(edges_3_);
    edge_list.push_back(edges_4_);

    //Invariants for nodes
    list<guard_d> invariant_0_;
    invariant_0_.emplace_back(0, logical_operator::less_equal, 2);
    
    list<guard_d> invariant_1_;
    invariant_1_.emplace_back(1, logical_operator::less_equal, 10);

    list<guard_d> invariant_2_;
    //invariant_2_.emplace_back(0, logical_operator::less_equal, 10);

    list<guard_d> invariant_3_;
    // invariant_3_.emplace_back(0, logical_operator::greater_equal, 10, 1);

    list<guard_d> invariant_4_;

    list<list<guard_d>> invariant_list;
    invariant_list.push_back(invariant_0_);
    invariant_list.push_back(invariant_1_);
    invariant_list.push_back(invariant_2_);
    invariant_list.push_back(invariant_3_);
    invariant_list.push_back(invariant_4_);

    // Guard List for edges
    list<guard_d> guard_0_;
    // guard_0_.emplace_back(0, logical_operator::less_equal, 2);
    
    list<guard_d> guard_1_;
    guard_1_.emplace_back(1, logical_operator::less, 2);

    list<guard_d> guard_2_;
    guard_2_.emplace_back(0, logical_operator::less, 1);

    list<guard_d> guard_3_;
    guard_3_.emplace_back(0, logical_operator::greater, 4);

    list<guard_d> guard_4_;
    guard_4_.emplace_back(1, logical_operator::less, 4);
    
    list<list<guard_d>> guard_list;
    guard_list.push_back(guard_0_);
    guard_list.push_back(guard_1_);
    guard_list.push_back(guard_2_);
    guard_list.push_back(guard_3_);
    guard_list.push_back(guard_4_);

    //Update list for edges
    list<update_d> update_0_;
    update_0_.emplace_back(1, 0);
    
    list<update_d> update_1_;
    // update_1_.emplace_back(0, 0);
    // update_1_.emplace_back(1,0);
    
    list<update_d> update_2_;
    update_2_.emplace_back(0, 0);

    list<update_d> update_3_;

    list<update_d> update_4_;

    list<list<update_d>> update_list;
    update_list.push_back(update_0_);
    update_list.push_back(update_1_);
    update_list.push_back(update_2_);
    update_list.push_back(update_3_);
    update_list.push_back(update_4_);

    // Timers
    int timer_count = 2;
    timer_d* timer_list;
    timer_list = (timer_d*)malloc(sizeof(timer_d) * timer_count);
    timer_list[0] = timer_d(0, 0);
    timer_list[1] = timer_d(1, 0);
    
    uneven_list<edge_d> node_to_edge(&edge_list, 5);
    uneven_list<guard_d> node_to_invariant(&invariant_list, 5);
    uneven_list<guard_d> edge_to_guard(&guard_list, 5);
    uneven_list<update_d> edge_to_update(&update_list, 5);

    // NOW ALLOCATE MEMORY ON DEVICE FOR ALL THIS SHIT!

    // uneven_list<edge_d>* node_to_edge_d = nullptr;
    // uneven_list<guard_d>* node_to_invariant_d = nullptr;
    // uneven_list<guard_d>* edge_to_guard_d = nullptr;
    // uneven_list<update_d>* edge_to_update_d = nullptr;
    //
    // timer_d* timers_d = nullptr;

    
    // cudaMalloc((void**)&node_to_edge_d, sizeof(uneven_list<edge_d>));
    // cudaMalloc((void**)&node_to_invariant_d, sizeof(uneven_list<guard_d>));
    // cudaMalloc((void**)&edge_to_guard_d, sizeof(uneven_list<guard_d>));
    // cudaMalloc((void**)&edge_to_update_d, sizeof(uneven_list<update_d>));
    // cudaMalloc((void**)&timers_d, sizeof(timer_d) * 2);
    //
    //
    // // Copy memory to device
    // node_to_edge.allocate_memory();
    // node_to_invariant.allocate_memory();
    // edge_to_guard.allocate_memory();
    // edge_to_update.allocate_memory();
    //
    // cudaMemcpy(node_to_edge_d, &node_to_edge, sizeof(uneven_list<edge_d>), cudaMemcpyHostToDevice);
    // cudaMemcpy(node_to_invariant_d, &node_to_invariant, sizeof(uneven_list<guard_d>), cudaMemcpyHostToDevice);
    // cudaMemcpy(edge_to_guard_d, &edge_to_guard, sizeof(uneven_list<guard_d>), cudaMemcpyHostToDevice);
    // cudaMemcpy(edge_to_update_d, &edge_to_update, sizeof(uneven_list<update_d>), cudaMemcpyHostToDevice);
    // cudaMemcpy(timers_d, timer_list, sizeof(timer_d) * 2, cudaMemcpyHostToDevice);

    //printf("yasss girl: %d %d %d %d\n", node_to_edge.max_elements_, node_to_edge.max_index_, node_to_edge_d->max_elements_, node_to_edge_d->max_index_);
    
    // const stochastic_model model(node_to_edge_d, node_to_invariant_d,
    //     edge_to_guard_d, edge_to_update_d, timers_d, 2);

    list<void*> free_list = list<void*>();
    const stochastic_model model(&node_to_edge, &node_to_invariant,
        &edge_to_guard, &edge_to_update, timer_list, timer_count);
    
    const cuda_simulator sim = cuda_simulator();
    simulation_strategy strategy = {
        32,
        512,
        100,
        1,
        20
    };
    sim.simulate(&model, &strategy, &free_list);

    free(timer_list);
    for(void* p : free_list)
    {
        cudaFree(p);
    }
    
    return 0;
}


