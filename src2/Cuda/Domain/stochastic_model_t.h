#pragma once

#ifndef STOCHASTIC_MODEL_T_H
#define STOCHASTIC_MODEL_T_H

#include "common.h"

class stochastic_model_t
{
private:
    node_t* start_node_;
    array_t<clock_timer_t*> timers_{0};
    
public:
    explicit stochastic_model_t(node_t* start_node, array_t<clock_timer_t*> timers);
    GPU array_t<clock_timer_t> create_internal_timers();
    GPU void reset_timers(array_t<clock_timer_t>* active_timers);
    GPU node_t* get_start_node() const;
    void cuda_allocate(stochastic_model_t** pointer, const allocation_helper* helper);
    void accept(visitor* v);
};

#endif