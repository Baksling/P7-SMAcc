
#include "Domain.h"
#include "common/sim_config.h"
#include "results/result_store.h"

class simulation_runner
{

public:

    static void simulate_gpu(const automata* model, sim_config* config);
    static void simulate_cpu(const automata* model, sim_config* config);
    
};
