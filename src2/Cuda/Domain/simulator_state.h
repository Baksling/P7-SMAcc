#ifndef SIMULATOR_STATE
#define SIMULATOR_STATE

#include "../common/macro.h"
#include "../Simulator/simulation_strategy.h"
#include "expressions/expression.h"
#include "../common/lend_array.h"
#include "clock_variable.h"
#include "stochastic_model_t.h"
#include "../common/cuda_stack.h"

#define MAX_DOUBLE (1.7976931348623158e+308)
#define NO_PROGRESS (-1.0)


//Prototype
class channel_medium;
class expression;
class result_manager;

class simulator_state
{
friend class result_manager;
    
private:
    GPU CPU double determine_progression(const node_t* node);

    unsigned int sim_id_ = 0;
    array_t<model_state> models_{0};
    array_t<clock_variable> variables_{nullptr, 0};
    array_t<clock_variable> timers_{nullptr, 0};
    
    double global_time_ = 0.0;
    double trace_time_ = 0.0;
    unsigned int steps_ = 0;

    CPU GPU void progress_timers(const double time);
    
    CPU GPU simulator_state(
        curandState* random,
        const cuda_stack<expression*>& expression_stack,
        const cuda_stack<double>& value_stack);
    
public:
    cuda_stack<double> value_stack{0};
    cuda_stack<expression*> expression_stack{0};
    curandState* random;

    CPU GPU lend_array<clock_variable> get_timers() const;
    CPU GPU lend_array<clock_variable> get_variables() const;

    CPU GPU void broadcast_channel(const model_state* current_state, const unsigned channel_id, const result_manager* results);
    CPU GPU void reset(unsigned sim_id, const stochastic_model_t* model);
    CPU GPU model_state* progress_sim(const model_options* options);

    //CONSTRUCTOR_METHOD
    CPU GPU static simulator_state init(
        const stochastic_model_t* multi_model,
        const model_options* options,
        curandState* random, void* memory_heap);
};

#endif
