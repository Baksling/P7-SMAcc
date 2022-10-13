#include "division_expression.h"

division_expression::division_expression(update_expression* left, update_expression* right)
{
    this->left_ = left;
    this->right_ = right;
}

void division_expression::evaluate(cuda_stack<double>* stack, lend_array<clock_timer_t>* timers,
    lend_array<system_variable>* variables)
{
    if(stack->count() < 2)
    {
        printf("stack not big enough to evaluate division expression");
        return;
    }
    
    const double left = stack->pop();
    const double right = stack->pop();

    if (right == 0.0)
    {
        printf("U fucking idiot. You divide by 0.. u stupid \n");
        printf("segmentation fault: hehe we did this \n");
    }
    const double result = left / right;
    
    stack->push(result);
}

void division_expression::accept(visitor* v)
{
    //TODO fix
    return;
}

update_expression* division_expression::cuda_allocate(allocation_helper* helper)
{
}

unsigned division_expression::get_depth() const
{
    const unsigned int left = this->left_->get_depth();
    const unsigned int right = this->right_->get_depth();

    return (left > right ? left : right) + 1;
}
