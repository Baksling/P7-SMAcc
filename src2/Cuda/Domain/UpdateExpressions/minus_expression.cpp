#include "minus_expression.h"

minus_expression::minus_expression(update_expression* left, update_expression* right)
{
    this->left_ = left;
    this->right_ = right;
}

void minus_expression::evaluate(cuda_stack<double>* stack, lend_array<clock_timer_t>* timers,
    lend_array<system_variable>* variables)
{
    if(stack->count() < 2)
    {
        printf("stack not big enough to evaluate minus expression");
        return;
    }
    
    const double left = stack->pop();
    const double right = stack->pop();
    const double result = left - right;
    
    stack->push(result);
}

void minus_expression::accept(visitor* v)
{
    //TODO fix
    return;
}

update_expression* minus_expression::cuda_allocate(allocation_helper* helper)
{
    
}

unsigned minus_expression::get_depth() const
{
    const unsigned int left = this->left_->get_depth();
    const unsigned int right = this->right_->get_depth();

    return (left > right ? left : right) + 1;
}
