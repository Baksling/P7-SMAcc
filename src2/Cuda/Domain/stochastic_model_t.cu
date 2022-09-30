#include "stochastic_model_t.h"

stochastic_model_t::stochastic_model_t(node_t* start_node, array_t<clock_timer_t>* timers)
{
    this->start_node_ = start_node;
    this->timers_ = timers;
}
