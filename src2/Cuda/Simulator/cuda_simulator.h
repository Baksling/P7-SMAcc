#pragma once

#ifndef CUDA_SIM_H
#define CUDA_SIM_H


//#include "stochastic_simulator.cu"
#include "../Domain/common.h"
#include "simulation_strategy.h"


class cuda_simulator
{
public:
    static void simulate(stochastic_model_t* model, simulation_strategy* strategy);
};

#endif
