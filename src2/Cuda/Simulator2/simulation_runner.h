#pragma once
#include "engine/model_oracle.h"
#include "common/sim_config.h"
#include "visitors/jit_compile_visitor.h"


class simulation_runner
{

public:

    static void simulate_gpu_aligned(const model_oracle* oracle, sim_config* config);
    static void simulate_gpu(const network* model, sim_config* config);
    static void simulate_cpu(const network* model, sim_config* config);

    static void simulate_gpu_jit(const network* model, jit_compile_visitor* optimizer, sim_config* config);
};
