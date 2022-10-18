
#ifndef STOCHASTIC_SIMULATOR_H
#define STOCHASTIC_SIMULATOR_H

#include "../common/macro.h"
#include "../Domain/stochastic_model_t.h"
#include "simulation_strategy.h"

class stochastic_simulator
{
public:
    static void simulate_gpu(const stochastic_model_t* model, simulation_strategy* strategy);
    static void simulate_cpu(stochastic_model_t* model, simulation_strategy* strategy);
};

#endif