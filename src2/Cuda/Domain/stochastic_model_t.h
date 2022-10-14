#pragma once

#ifndef STOCHASTIC_MODEL_T_H
#define STOCHASTIC_MODEL_T_H

#include "common.h"

class stochastic_model_t
{
private:
    node_t* start_node_;
    array_t<clock_timer_t*> timers_{0};
    array_t<system_variable*> variables_{0};
    
public:
    explicit stochastic_model_t(node_t* start_node, array_t<clock_timer_t*> timers, array_t<system_variable*> variables);

    //SIMULATOR METHODS
    CPU GPU array_t<clock_timer_t> create_internal_timers() const;
    CPU GPU array_t<system_variable> create_internal_variables() const;
    CPU GPU void reset_timers(const array_t<clock_timer_t>* active_timers, const array_t<system_variable>* active_variables) const;
    CPU GPU node_t* get_start_node() const;


    //HOST METHODS
    void cuda_allocate(stochastic_model_t** pointer, const allocation_helper* helper) const;
    void accept(visitor* v) const;
};

#endif