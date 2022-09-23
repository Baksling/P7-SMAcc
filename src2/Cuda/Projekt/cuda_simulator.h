#pragma once

#ifndef CUDA_SIMULATOR_H
#define CUDA_SIMULATOR_H

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
    cuda_simulator();
    __host__ void simulate_2(uneven_list<edge_d> *node_to_edge, uneven_list<guard_d> *node_to_invariant, uneven_list<guard_d> *edge_to_guard, uneven_list<update_d> *edge_to_update, int timer_amount, timer_d* timers) const;
};

#endif // CUDA_SIMULATOR_H