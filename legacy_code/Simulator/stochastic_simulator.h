
#ifndef STOCHASTIC_SIMULATOR_H
#define STOCHASTIC_SIMULATOR_H

#include "./writers/result_writer.h"
#include "../common/macro.h"
#include "../Domain/stochastic_model_t.h"
#include "simulation_strategy.h"

class stochastic_simulator
{
private:

    static model_options build_options(stochastic_model_t* model, const simulation_strategy* strategy); 
    
public:
    static void simulate_gpu(stochastic_model_t* model,
        const simulation_strategy* strategy,
        result_writer* r_writer,
        const bool verbose);
    
    static void simulate_cpu(stochastic_model_t* model,
        const simulation_strategy* strategy,
        result_writer* r_writer,
        const bool verbose);
};

#endif