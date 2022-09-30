#pragma once

#ifndef STOCHASTIC_MODEL_T_H
#define STOCHASTIC_MODEL_T_H

#include "node_t.h"

class stochastic_model_t
{
private:
    node_t* start_node_;
    array_t<clock_timer_t>* timers_;
public:
    explicit stochastic_model_t(node_t* start_node, array_t<clock_timer_t>* timers);
};

#endif