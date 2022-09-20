#ifndef CUDASIMULATOR_H
#define CUDASIMULATOR_H

#include "node_d.h"
#include "edge_d.h"
#include "guard_d.h"
#include "update_d.h"
#include "timer_d.h"
#include "uneven_list.h"
#include <cuda.h>
#include <cuda_runtime.h>

class cuda_simulator
{
    array_info<node_d>* nodes_;
    array_info<edge_d>* edges_;
    array_info<guard_d>* guards_;
    array_info<update_d>* updates_;
    array_info<timer_d>* timers_;
public:
    cuda_simulator(array_info<node_d>* nodes, array_info<edge_d>* edges, array_info<guard_d>* guards, array_info<update_d>* updates, array_info<timer_d>* timers);
    __host__ void simulate(int max_nr_of_steps);
};

#endif
