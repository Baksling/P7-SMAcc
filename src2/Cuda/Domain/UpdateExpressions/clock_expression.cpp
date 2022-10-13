#include "clock_expression.h"

clock_expression::clock_expression(int clock_id)
{
    this->clock_id_ = clock_id; 
}

void clock_expression::evaluate(cuda_stack<double>* stack, lend_array<clock_timer_t>* timers,
    lend_array<system_variable>* variables)
{
    const clock_timer_t* timer = timers->at(this->clock_id_);

    if (timer == nullptr) printf("segmentation fault: hehehe timer is a nullptr\n");
    
    const double value = timer->get_time();    
    stack->push(value);
}

void clock_expression::accept(visitor* v)
{
    //TODO fix
}

update_expression* clock_expression::cuda_allocate(allocation_helper* helper)
{
}

unsigned clock_expression::get_depth() const
{
    return 1;
}
