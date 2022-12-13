#pragma once

#include "../common/macro.h"
#include "../Domain/stochastic_model_t.h"
#include "simulation_strategy.h"
#include "writers/result_manager.h"


class stochastic_engine
{
private:

public:
    static bool run_gpu(const stochastic_model_t* model,
                        const model_options* options,
                        const result_manager* output,
                        const simulation_strategy* strategy,
                        void* total_memory_heap);

    static bool run_cpu(const stochastic_model_t* model,
                        const model_options* options,
                        result_manager* output,
                        const simulation_strategy* strategy,
                        void* total_memory_heap);
};
