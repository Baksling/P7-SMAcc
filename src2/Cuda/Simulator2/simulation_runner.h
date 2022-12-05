
#include "engine/model_oracle.h"
#include "common/sim_config.h"


class simulation_runner
{

public:

    static void simulate_oracle(const model_oracle* oracle, sim_config* config);
    static void simulate_gpu(const network* model, sim_config* config);
    static void simulate_cpu(const network* model, sim_config* config);
    
};
