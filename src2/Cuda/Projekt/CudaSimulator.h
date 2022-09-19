#ifndef CUDASIMULATOR_H
#define CUDASIMULATOR_H

#include "../../Node.h"
#include "../../Edge.h"
#include "../../Timer.h"
#include "../../Guard.h"
#include "../../Update.h"

template <typename T>
struct array_info
{
    T* arr;
    int size;
};

class cuda_simulator
{
    array_info<node>* nodes_;
    array_info<edge>* edges_;
    array_info<guard>* guards_;
    array_info<update>* updates_;
    array_info<timer>* timers_;
public:
    cuda_simulator(array_info<node>* nodes, array_info<edge>* edges, array_info<guard>* guards, array_info<update>* updates, array_info<timer>* timers);
    __host__ void simulate(int max_nr_of_steps);
};

#endif
