#pragma once

#ifndef EDGE_T_H
#define EDGE_T_H

#include <list>
#include "../common/macro.h"
#include "../common/array_t.h"
#include "../common/allocation_helper.h"
#include "constraint_t.h"
#include "update_t.h"
#include "simulator_state.h"
#include "../Visitors/visitor.h"


class edge_t
{
private:
    int id_;
    float weight_;
    node_t* dest_;
    array_t<constraint_t*> guards_{0};
    array_t<update_t*> updates_{0};
    explicit edge_t(edge_t* source, node_t* dest, array_t<constraint_t*> guard, array_t<update_t*> updates);
public:
    explicit edge_t(int id, float weight, node_t* dest, array_t<constraint_t*> guard, array_t<update_t*> updates);

    //SIMULATION METHODS
    GPU CPU node_t* get_dest() const;
    CPU GPU bool evaluate_constraints(simulator_state* state) const;
    CPU GPU void execute_updates(simulator_state* state) const;
    CPU GPU float get_weight() const;


    //HOST METHODS
    int get_id() const;
    void accept(visitor* v) const;
    void cuda_allocate(edge_t** pointer, const allocation_helper* helper);
    void cuda_allocate_2(edge_t* cuda_p, const allocation_helper* helper);
    int get_updates_size() const;
};

#endif
