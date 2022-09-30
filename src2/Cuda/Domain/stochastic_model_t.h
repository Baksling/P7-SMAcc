﻿#pragma once

#include "node_t.h"

class stochastic_model_t : public element
{
private:
    node_t* start_node_;
    array_t<timer_t>* timers_;
public:
    explicit stochastic_model_t(node_t* start_node, array_t<timer_t>* timers);
    void accept(visistor& v) override;
};
