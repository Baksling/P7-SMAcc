
#ifndef STOCHASTIC_SIMULATOR_H
#define STOCHASTIC_SIMULATOR_H

#include "result_writer.h"
#include "../common/macro.h"
#include "../Domain/stochastic_model_t.h"
#include "simulation_strategy.h"

class stochastic_simulator
{
private:

    static model_options build_options(const stochastic_model_t* model, const simulation_strategy* strategy); 
    
public:
    static void simulate_gpu(const stochastic_model_t* model, const simulation_strategy* strategy, const result_writer* r_writer);
    static void simulate_cpu(const stochastic_model_t* model, const simulation_strategy* strategy);
};

#endif