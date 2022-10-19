#pragma once

#ifndef STOCHASTIC_MODEL_T_H
#define STOCHASTIC_MODEL_T_H

#include "../common/macro.h"
#include "../common/array_t.h"
#include "../common/allocation_helper.h"
#include "clock_variable.h"
#include "../Visitors/visitor.h"
#include "node_t.h"

class stochastic_model_t
{
private:
    node_t* start_node_;
    array_t<clock_variable*> timers_{0};
    array_t<clock_variable*> variables_{0};
    
public:
    explicit stochastic_model_t(node_t* start_node, array_t<clock_variable*> timers, array_t<clock_variable*> variables);

    //SIMULATOR METHODS
    CPU GPU array_t<clock_variable> create_internal_timers() const;
    CPU GPU array_t<clock_variable> create_internal_variables() const;
    CPU GPU void reset_timers(const array_t<clock_variable>* active_timers, const array_t<clock_variable>* active_variables) const;
    CPU GPU node_t* get_start_node() const;


    //HOST METHODS
    void cuda_allocate(stochastic_model_t** pointer, const allocation_helper* helper) const;
    unsigned int get_variable_count() const; 
    void accept(visitor* v) const;
};

#endif