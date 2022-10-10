#pragma once

#ifndef CPU_SIMULATOR_H
#define CPU_SIMULATOR_H

#include "stochastic_simulator.cu"
#include "simulation_strategy.h"

class cpu_simulator
{
public:
    static void simulate(stochastic_model_t* model, const simulation_strategy* strategy);
};

#endif