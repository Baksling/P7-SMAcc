#pragma once
#include "common/io_path.h"
#include "engine/model_oracle.h"
#include "common/sim_config.h"
#include "visitors/expr_compiler_visitor.h"


class simulation_runner
{

public:

    static void simulate_gpu_aligned(const model_oracle* oracle, sim_config* config, const io_path* paths);
    static void simulate_gpu(const network* model, sim_config* config, const io_path* paths);
    static void simulate_cpu(const network* model, sim_config* config, const io_path* paths);

    static void simulate_gpu_jit(const network* model, expr_compiler_visitor* optimizer, sim_config* config, io_path* paths);
};
