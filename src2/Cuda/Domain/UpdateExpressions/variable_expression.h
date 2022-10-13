#pragma once

#ifndef VARIABLE_EXPRESSION_H
#define VARIABLE_EXPRESSION_H


#include "update_expression.h"

class variable_expression final : public update_expression
{
private:
    int variable_id_;
public:
    explicit variable_expression(int variable_id);

    //SIMULATION methods
    void evaluate(cuda_stack<double>* stack, lend_array<clock_timer_t>* timers, lend_array<system_variable>* variables) override;
    void push_children(cuda_stack<update_expression*> stack) override;

    //HOST methods
    void accept(visitor* v) override;
    update_expression* cuda_allocate(allocation_helper* helper) override;
    unsigned int get_depth() const override;
};

#endif