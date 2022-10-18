#pragma once
#ifndef NODE_T_H
#define NODE_T_H

#include "../common/macro.h"
#include "../common/array_t.h"
#include "../common/lend_array.h"
#include "../common/allocation_helper.h"

#include "expressions/expression.h"
#include "constraint_t.h"
#include "simulator_state.h"

#include "../Visitors/visitor.h"


class edge_t;

class node_t final
{
private:
    int id_;
    bool is_goal_;
    bool is_branch_point_;
    expression* lambda_expression_;
    array_t<constraint_t*> invariants_{0};
    array_t<edge_t*> edges_{0};
    explicit node_t(node_t* source, array_t<constraint_t*> invariant, array_t<edge_t*> edges, expression* lambda); 
public:
    explicit node_t(int id, array_t<constraint_t*> invariants,
        bool is_branch_point = false, bool is_goal = false, expression* lambda = nullptr);

    //SIMULATOR METHODS
    GPU CPU int get_id() const;
    GPU CPU double get_lambda(simulator_state* state) const;
    CPU GPU lend_array<edge_t*> get_edges();
    CPU GPU bool is_goal_node() const;
    CPU GPU bool evaluate_invariants(const simulator_state* state) const;
    CPU GPU bool max_time_progression(const lend_array<clock_variable>* timers, double* out_max_progression) const;
    CPU GPU bool is_branch_point() const;

    //HOST METHODS
    void set_edges(std::list<edge_t*>* list);
    void accept(visitor* v) const;
    void pretty_print() const;
    void cuda_allocate(node_t** pointer, const allocation_helper* helper);
    void cuda_allocate_2(node_t* cuda_p, const allocation_helper* helper) const;
};

#endif

