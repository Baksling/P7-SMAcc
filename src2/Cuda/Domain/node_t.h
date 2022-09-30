#pragma once
#ifndef NODE_T_H
#define NODE_T_H

#include "common.h"
#include "constraint_t.h"
#include "edge_t.h"
class edge_t;

class node_t
{
private:
    int id_;
    bool is_goal_;
    constraint_t* invariant_;
    array_t<edge_t> edges_{0};
public:
    explicit node_t(int id, constraint_t* invariant = nullptr, bool is_goal = false);
    GPU CPU int get_id() const;
    void set_edges(std::list<edge_t>* list);
    GPU lend_array<edge_t> get_edges();
    GPU bool is_goal_node() const;
    GPU bool evaluate_invariants(const lend_array<clock_timer_t>* timers) const;
    GPU double max_time_progression(const lend_array<clock_timer_t>* timers, double max_progression = 100.0) const;
};

#endif

