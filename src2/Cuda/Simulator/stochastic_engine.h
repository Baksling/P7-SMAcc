#pragma once

#include "../common/macro.h"
#include "../Domain/stochastic_model_t.h"
#include "simulation_strategy.h"

class stochastic_engine
{
private:

public:
    static bool run_gpu(const stochastic_model_t* model,
                        const model_options* options,
                        const simulation_result_container* output, const simulation_strategy* strategy);

    static bool run_cpu(const stochastic_model_t* model,
                        const model_options* options,
                        simulation_result_container* output, const simulation_strategy* strategy);
};
