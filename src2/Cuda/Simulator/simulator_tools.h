﻿#ifndef SIMULATOR_TOOLS_H
#define SIMULATOR_TOOLS_H

#include "../common/macro.h"
#include "../common/lend_array.h"
#include <map>
#include "simulation_result.h"
#include "simulation_strategy.h"
#include "../common/allocation_helper.h"
#include "../Domain/edge_t.h"
#include "../Domain/channel_medium.h"

class simulator_tools
{
public:
    CPU GPU static bool bit_is_set(const unsigned long long* n, unsigned int i);
    CPU GPU static void set_bit(unsigned long long* n, unsigned int i);
    CPU GPU static void unset_bit(unsigned long long* n, unsigned int i);
    

    CPU GPU static edge_t* choose_next_edge_bit(
        simulator_state* state,
        const lend_array<edge_t*>* edges,
        curandState* r_state);

    
    CPU GPU static  edge_t* choose_next_edge(
        simulator_state* state,
        const lend_array<edge_t*>* edges,
        curandState* r_state);
};


#endif