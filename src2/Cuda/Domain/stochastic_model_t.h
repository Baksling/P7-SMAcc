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

    //SIMULATOR METHODS
    CPU GPU array_t<clock_timer_t> create_internal_timers() const;
    CPU GPU void reset_timers(array_t<clock_timer_t>* active_timers) const;
    CPU GPU node_t* get_start_node() const;


    //HOST METHODS
    void cuda_allocate(stochastic_model_t** pointer, const allocation_helper* helper);
    void accept(visitor* v) const;
};

#endif