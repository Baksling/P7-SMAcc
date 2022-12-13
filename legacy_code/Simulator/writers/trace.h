#pragma once

#include "../../common/lend_array.h"

struct trace_interval
{
    enum interval_type
    {
        disabled,
        time_interval,
        step_interval
    } mode;
    double value;

    
};


struct trace_vector
{
    unsigned step;
    unsigned item_id;
    double value;
    bool is_node;
};

struct trace_pointers
{
private:
    const bool owns_pointers_;
public:
    explicit trace_pointers(const bool owns_pointers,
        unsigned simulations,
        unsigned size,
        unsigned* stack_counters, trace_vector* data);
    const unsigned size;
    const unsigned simulations;
    
    unsigned* stack_counters;
    trace_vector* data;

    lend_array<trace_vector> get_trace(unsigned sim_id) const;
    
    void free_internals() const;
};
