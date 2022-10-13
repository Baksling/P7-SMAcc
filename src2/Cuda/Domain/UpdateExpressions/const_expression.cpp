#include "const_expression.h"

const_expression::const_expression(double value)
{
    this->value_ = value;
}

void const_expression::evaluate(cuda_stack<double>* stack, lend_array<clock_timer_t>* timers,
    lend_array<system_variable>* variables)
{
    stack->push(this->value_);
}

void const_expression::accept(visitor* v)
{
    //TODO fix
    return;
}

update_expression* const_expression::cuda_allocate(allocation_helper* helper)
{
}

unsigned const_expression::get_depth() const
{
    return 1;
}
