#include "trace.h"


trace_pointers::trace_pointers(const bool owns_pointers,
    const unsigned simulations,
    const unsigned size,
    unsigned* stack_counters,
    trace_vector* data)
    : owns_pointers_(owns_pointers), size(size), simulations(simulations)
{
    this->stack_counters = stack_counters;
    this->data = data;
}

lend_array<trace_vector> trace_pointers::get_trace(const unsigned sim_id) const
{
    return lend_array<trace_vector>(&this->data[sim_id*size], this->stack_counters[sim_id]);
}

void trace_pointers::free_internals() const
{
    if(!owns_pointers_) return;
    free(this->stack_counters);
    free(this->data);
}