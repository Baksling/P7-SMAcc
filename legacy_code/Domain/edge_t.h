#pragma once

#ifndef EDGE_T_H
#define EDGE_T_H

#include <list>
#include "../common/macro.h"
#include "../common/array_t.h"
#include "../common/allocation_helper.h"
#include "expressions/constraint_t.h"
#include "update_t.h"
#include "simulator_state.h"
#include "../Visitors/visitor.h"

#define NO_CHANNEL (0xffffffff)

struct edge_channel
{
    bool is_listener;
    unsigned channel_id;
};


class edge_t
{
private:
    int id_;
    edge_channel channel_{};
    expression* weight_expression_;
    node_t* dest_;
    array_t<constraint_t> guards_{0};
    array_t<update_t> updates_{0};
public:
    explicit edge_t(int id,
                    expression* weight_expression,
                    node_t* dest,
                    const array_t<constraint_t>& guard,
                    const array_t<update_t>& updates,
                    edge_channel channel = { true, NO_CHANNEL });

    //SIMULATION METHODS
    GPU CPU node_t* get_dest() const;
    CPU GPU bool evaluate_constraints(simulator_state* state) const;
    CPU GPU void execute_updates(simulator_state* state) const;
    CPU GPU double get_weight(simulator_state* state) const;
    CPU GPU unsigned get_channel() const; //may return NO_CHANNEl for no channels
    CPU GPU bool is_listener() const;
    CPU GPU int get_id() const;

    //HOST METHODS
    void accept(visitor* v) const;
    void pretty_print(std::ostream& os) const;
    void cuda_allocate(edge_t* pointer, allocation_helper* helper) const;
    int get_updates_size() const;
};

#endif
