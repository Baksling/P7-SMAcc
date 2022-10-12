#pragma once
#ifndef NODE_T_H
#define NODE_T_H

#include "common.h"

class edge_t;

class node_t final
{
private:
    int id_;
    float ex_lambda_;
    bool is_goal_;
    bool is_branch_point_;
    array_t<constraint_t*> invariants_{0};
    array_t<edge_t*> edges_{0};
    explicit node_t(node_t* source, array_t<constraint_t*> invariant, array_t<edge_t*> edges); 
public:
    explicit node_t(int id, array_t<constraint_t*> invariants,
        bool is_branch_point = false, bool is_goal = false, float ex_lambda = 1.0);

    //SIMULATOR METHODS
    GPU CPU int get_id() const;
    GPU CPU float get_lambda() const;
    CPU GPU lend_array<edge_t*> get_edges();
    CPU GPU bool is_goal_node() const;
    CPU GPU bool evaluate_invariants(const lend_array<clock_timer_t>* timers) const;
    CPU GPU double max_time_progression(const lend_array<clock_timer_t>* timers, double max_progression = 100.0) const;
    CPU GPU bool is_branch_point() const;

    //HOST METHODS
    void set_edges(std::list<edge_t*>* list);
    void accept(visitor* v) const;
    void cuda_allocate(node_t** pointer, const allocation_helper* helper);
    void cuda_allocate_2(node_t* cuda_p, const allocation_helper* helper) const;
};

#endif

