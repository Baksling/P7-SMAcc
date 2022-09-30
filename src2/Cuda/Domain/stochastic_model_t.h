#pragma once
#include "node_t.h"

class stochastic_model_t
{
private:
    node_t* start_node_;
    array_t<timer_t>* timers_;
public:
    explicit stochastic_model_t(node_t* start_node, array_t<timer_t>* timers);
    array_t<timer_t> create_internal_timers() const;
    void reset_timers(array_t<timer_t>* active_timers) const;
    node_t* get_start_node() const; 
};
