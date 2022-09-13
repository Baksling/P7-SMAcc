﻿#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "Node.h"
#include "Edge.h"
#include "Timer.h"
#include <map>

class simulator
{
private:
    int max_steps_;
    map<int, timer*> timers_;
public:
    simulator(int max_steps = 1000);
    bool simulate(node* start_node, int n_step = 0);
    void add_timer(int id);
    timer* get_timer(int id);
};

#endif // SIMULATOR_H
