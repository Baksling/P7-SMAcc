#pragma once

#include "common.h"
#include "constraint_t.h"
#include "edge_t.h"

class node_t final : public element
{
private:
    int id_;
    bool is_goal_;
    bool is_branch_;    
    constraint_t* invariant_;
    array_t<edge_t> edges_{0};
public:
    explicit node_t(int id, constraint_t* invariant = nullptr, bool is_goal = false, bool is_branch = false);
    void set_edges(std::list<edge_t>* list);
    GPU lend_array<edge_t> get_edges();
    CPU GPU bool is_goal_node() const;
    GPU bool evaluate_invariants(const lend_array<timer_t>* timers) const;
    void accept(visistor& v) override;
    int get_id() const;
    CPU GPU bool is_branch() const;
};

