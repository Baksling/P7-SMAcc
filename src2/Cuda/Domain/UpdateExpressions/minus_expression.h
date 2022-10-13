#pragma once

#ifndef MINUS_EXPRESSION_H
#define MINUS_EXPRESSION_H

#include "update_expression.h"

class minus_expression : public update_expression
{
private:
    update_expression* left_;
    update_expression* right_;
public:
    explicit minus_expression(update_expression* left, update_expression* right);

    //SIMULATION methods
    void evaluate(cuda_stack<double>* stack, lend_array<clock_timer_t>* timers, lend_array<system_variable>* variables) override;
    void push_children(cuda_stack<update_expression*> stack) override;

    
    //HOST methods
    void accept(visitor* v) override;
    update_expression* cuda_allocate(allocation_helper* helper) override;
    unsigned int get_depth() const override;
};


#endif