#pragma once
#ifndef NODE_T_H
#define NODE_T_H

#include "../common/macro.h"
#include "../common/array_t.h"
#include "../common/lend_array.h"
#include "../common/allocation_helper.h"

#include "expressions/expression.h"
#include "expressions/constraint_t.h"
#include "edge_t.h"

#include "../Visitors/visitor.h"


class simulator_state;

class node_t final
{
private:
    int id_;
    bool is_goal_;
    bool is_branch_point_;
    expression* lambda_expression_;
    array_t<constraint_t> invariants_{0};
    array_t<edge_t> edges_{0};
    explicit node_t(const node_t* source, const array_t<constraint_t>& invariant, const array_t<edge_t>& edges, expression* lambda); 
public:
    explicit node_t(int id, const array_t<constraint_t>& invariants,
        bool is_branch_point = false, bool is_goal = false, expression* lambda = nullptr);

    //SIMULATOR METHODS
    GPU CPU int get_id() const;
    GPU CPU double get_lambda(simulator_state* state) const;
    CPU GPU lend_array<edge_t> get_edges();
    CPU GPU bool is_goal_node() const;
    CPU GPU bool evaluate_invariants(simulator_state* state) const;
    CPU GPU bool max_time_progression(simulator_state* state, double* out_max_progression) const;
    CPU GPU bool is_branch_point() const;
    CPU GPU bool is_progressible() const;

    //HOST METHODS
    void set_edges(std::list<edge_t>* list);
    void accept(visitor* v) const;
    void pretty_print() const;
    void cuda_allocate(node_t* pointer, allocation_helper* helper);
};

#endif

