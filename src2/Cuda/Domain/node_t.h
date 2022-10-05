#pragma once
#ifndef NODE_T_H
#define NODE_T_H

#include "common.h"

class edge_t;

class node_t final
{
private:
    int id_;
    bool is_goal_;
    bool is_branch_point_;
    constraint_t* invariant_;
    array_t<edge_t*> edges_{0};
    explicit node_t(node_t* source, constraint_t* invariant, array_t<edge_t*> edges); 
public:
    explicit node_t(int id, bool is_branch_point = false, constraint_t* invariant = nullptr, bool is_goal = false);
    GPU CPU int get_id() const;
    void set_edges(std::list<edge_t*>* list);
    CPU GPU lend_array<edge_t*> get_edges();
    CPU GPU bool is_goal_node() const;
    GPU bool evaluate_invariants(const lend_array<clock_timer_t>* timers) const;
    GPU double max_time_progression(const lend_array<clock_timer_t>* timers, double max_progression = 100.0) const;
    CPU GPU bool is_branch_point() const;
    void accept(visitor* v);
    void cuda_allocate(node_t** pointer, std::list<void*>* free_list);
};

#endif

