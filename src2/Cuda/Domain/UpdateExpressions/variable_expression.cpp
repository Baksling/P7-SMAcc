#include "variable_expression.h"

variable_expression::variable_expression(int variable_id)
{
    this->variable_id_ = variable_id;
}

void variable_expression::evaluate(cuda_stack<double>* stack, lend_array<clock_timer_t>* timers,
    lend_array<system_variable>* variables)
{
    const system_variable* p = variables->at(this->variable_id_);

    if (p == nullptr) printf("segmentation fault: hehehe variable is a nullptr\n");
    
    const int value = p->get_value();    
    stack->push(value);
}

void variable_expression::accept(visitor* v)
{
}

update_expression* variable_expression::cuda_allocate(allocation_helper* helper)
{
}

unsigned variable_expression::get_depth() const
{
    return 1;
}
