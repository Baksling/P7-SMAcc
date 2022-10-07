#pragma once

#ifndef EDGE_T_H
#define EDGE_T_H

#include "common.h"
#include <list>

class node_t;

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
    explicit edge_t(int id, float weight, node_t* dest, array_t<constraint_t*> guard);
    CPU GPU float get_weight() const;
    GPU CPU node_t* get_dest() const;
    void set_updates(std::list<update_t*>* updates);
    GPU bool evaluate_constraints(const lend_array<clock_timer_t>* timers) const;
    GPU void execute_updates(const lend_array<clock_timer_t>* timers);
    void accept(visitor* v) const;
    int get_id() const;
    void cuda_allocate(edge_t** pointer, const allocation_helper* helper);
    void cuda_allocate_2(edge_t* cuda_p, const allocation_helper* helper);
};

#endif
