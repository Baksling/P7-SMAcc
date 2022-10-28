﻿#ifndef SIMULATOR_STATE
#define SIMULATOR_STATE

#include "../common/macro.h"
#include "expressions/expression.h"
#include "../common/lend_array.h"
#include "clock_variable.h"
#include "stochastic_model_t.h"
#include "../common/cuda_stack.h"
#include "../Simulator/simulation_result.h"
#include "../Simulator/simulation_strategy.h"


#define MAX_DOUBLE (1.7976931348623158e+308)

struct model_state
{
    node_t* current_node;
    bool reached_goal;
};

//Prototype
class channel_medium;
class expression;

class simulator_state
{
private:
    GPU CPU double determine_progression(const node_t* node, curandState* r_state);

    unsigned int sim_id_ = 0;
    
    array_t<model_state> models_{0};
    array_t<clock_variable> variables_{nullptr, 0};
    array_t<clock_variable> timers_{nullptr, 0};
    
    double global_time_ = 0.0;
    unsigned int steps_ = 0;

    CPU GPU void progress_timers(const double time);
    
    CPU GPU simulator_state(
        const cuda_stack<expression*>& expression_stack,
        const cuda_stack<double>& value_stack,
        channel_medium* medium);
    
public:
    cuda_stack<double> value_stack{0};
    cuda_stack<expression*> expression_stack{0};
    channel_medium* medium;

    CPU GPU lend_array<clock_variable> get_timers() const;
    CPU GPU lend_array<clock_variable> get_variables() const;
    
    CPU GPU void reset(unsigned sim_id, const stochastic_model_t* model);
    CPU GPU model_state* progress_sim(const model_options* options, curandState* r_state);
    CPU GPU double evaluate_expression(expression* expr);

    CPU GPU void write_result(simulation_result* output_array) const;
    CPU GPU void free_internals();

    //CONSTRUCTOR_METHOD
    CPU GPU static simulator_state from_multi_model(const stochastic_model_t* multi_model, const model_options* options);
};

#endif
