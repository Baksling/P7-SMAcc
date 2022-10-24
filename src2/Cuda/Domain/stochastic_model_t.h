#pragma once
#include "stochastic_model_t.h"
#include "simulator_state.h"
#include "../common/array_t.h"
#include "node_t.h"


class stochastic_model_t
{
friend class simulator_state;
    
private:
    array_t<node_t> models_{0};
    array_t<clock_variable> timers_{0};
    array_t<clock_variable> variables_{0};
public:
    explicit stochastic_model_t(const array_t<node_t> models,
                           const array_t<clock_variable> timers,
                           const array_t<clock_variable> variables);

    //Simulation methods
    //None :)
    
    //Host methods
    unsigned int get_variable_count() const;
    void cuda_allocate(stochastic_model_t* device, const allocation_helper* helper) const;
    void accept(visitor* v) const;
void pretty_print() const;
};
