#pragma once
#include <curand_kernel.h>

#include "edge_d.h"
#include "guard_d.h"
#include "timer_d.h"
#include "uneven_list.h"
#include "update_d.h"


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
        uneven_list<edge_d>* node_to_edge_;
        uneven_list<guard_d>* node_to_invariant_;
        uneven_list<guard_d>* edge_to_guard_;
        uneven_list<update_d>* edge_to_update_;


public:
        stochastic_model(
                uneven_list<edge_d>* node_to_edge,
                uneven_list<guard_d>* node_to_invariant,
                uneven_list<guard_d>* edge_to_guard,
                uneven_list<update_d>* edge_to_update,
                timer_d* timers, int timer_count);

        //array methods
        GPU array_info<edge_d> get_node_edges(int node_id) const;
        GPU array_info<guard_d> get_node_invariants(int node_id) const;
        GPU array_info<guard_d> get_edge_guards(int edge_id) const;
        GPU array_info<update_d> get_updates(int edge_id) const;
        GPU void traverse_edge_update(int edge_id, const array_info<timer_d>* local_timers) const;

        //state functions
        GPU int get_start_node() const;
        GPU bool is_goal_node(int node_id) const;
        GPU array_info<timer_d> copy_timers() const;
        GPU void reset_timers(const array_info<timer_d>* timers) const;

        //allocation
        void cuda_allocate(stochastic_model** p) const;
};
