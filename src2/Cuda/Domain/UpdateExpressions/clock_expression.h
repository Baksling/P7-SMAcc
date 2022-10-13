#ifndef CLOCK_EXPRESSION_H
#define CLOCK_EXPRESSION_H


#include "update_expression.h"

class clock_expression final : public update_expression
{
private:
    int clock_id_;
public:
    explicit clock_expression(int clock_id);

    //SIMULATION methods
    void evaluate(cuda_stack<double>* stack, lend_array<clock_timer_t>* timers, lend_array<system_variable>* variables) override;
    void push_children(cuda_stack<update_expression*> stack) override;

    
    //HOST method
    void accept(visitor* v) override;
    update_expression* cuda_allocate(allocation_helper* helper) override;
    unsigned int get_depth() const override;
};

#endif