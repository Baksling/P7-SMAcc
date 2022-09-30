#pragma once
#include <curand_kernel.h>

#include "constraint_d.h"
#include "edge_d.h"
#include "guard_d.h"
#include "timer_d.h"
#include "uneven_list.h"
#include "update_d.h"
#include "cuda_map.h"


struct random_state
{
        curandState* state;
        const unsigned long seed; 
};

class stochastic_model
{
private:
        int timer_count_;
        timer_d* timers_;
        int constraint_count_;
        constraint_d* constraints_;
        uneven_list<edge_d>* node_to_edge_;
        cuda_map<int, int>* node_to_invariant_;
        cuda_map<int, int>* edge_to_guard_;
        uneven_list<update_d>* edge_to_update_;


public:
        stochastic_model(
                uneven_list<edge_d>* node_to_edge,
                cuda_map<int, int>* node_to_invariants,
                cuda_map<int, int>* edge_to_guards,
                uneven_list<update_d>* edge_to_update,
                constraint_d* constraints,
                int constraint_count,
                timer_d* timers,
                int timer_count);

        //array methods
        GPU array_info<edge_d> get_node_edges(int node_id) const;
        GPU constraint_d* get_node_invariants(int node_id) const;
        GPU constraint_d* get_edge_guards(int edge_id) const;
        GPU array_info<update_d> get_updates(int edge_id) const;
        GPU void traverse_edge_update(int edge_id, const array_info<timer_d>* local_timers) const;

        //state functions
        GPU array_info<constraint_d> get_constraints() const;
        GPU int get_start_node() const;
        GPU bool is_goal_node(int node_id) const;
        GPU array_info<timer_d> copy_timers() const;
        GPU void reset_timers(const array_info<timer_d>* timers) const;

        //allocation
        void cuda_allocate(stochastic_model** p, list<void*>* free_list) const;

        //misc
        CPU GPU void pretty_print() const;
};
