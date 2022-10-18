#ifndef SIMULATOR_STATE
#define SIMULATOR_STATE

#include "../common/macro.h"
#include "expressions/expression.h"
#include "../common/lend_array.h"
#include "clock_variable.h"
#include "../common/cuda_stack.h"

//Prototype
class expression;

class simulator_state
{
public:
    cuda_stack<double> value_stack;
    cuda_stack<expression*> expression_stack;
    lend_array<clock_variable> variables{nullptr, 0};
    lend_array<clock_variable> timers{nullptr, 0};

    CPU GPU double evaluate_expression(expression* expr);
};

#endif
