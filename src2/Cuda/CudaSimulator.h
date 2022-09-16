#ifndef CUDASIMULATOR_H
#define CUDASIMULATOR_H

#include "../Node.h"
#include "../Edge.h"
#include "../Timer.h"
#include "../Guard.h"
#include "../Update.h"

struct array_info
{
    void* arr;
    int size;
};

class cuda_simulator
{
    node* nodes_;
    edge* edges_;
    guard* guards_;
    update* updates_;
    timer* timers_;
public:
    cuda_simulator(array_info nodes, array_info edges, array_info guards, array_info updates, array_info timers);
    void simulate(int max_nr_of_steps);
};

#endif
