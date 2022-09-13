#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "Node.h"
#include "Edge.h"

class simulator
{
private:
    int max_steps_;
public:
    simulator(int max_steps = 1000);
    bool simulate(node* start_node, int n_step = 0);
    
};

#endif // SIMULATOR_H
