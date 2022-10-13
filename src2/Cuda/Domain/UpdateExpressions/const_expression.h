#pragma once

#ifndef CONST_EXPRESSION_H
#define CONST_EXPRESSION_H


#include "update_expression.h"

class const_expression final : public update_expression
{
private:
    double value_;
public:
    explicit const_expression(double value);

    //SIMULATION methods
    void evaluate(cuda_stack<double>* stack, lend_array<clock_timer_t>* timers, lend_array<system_variable>* variables) override;

    //HOST methods
    void accept(visitor* v) override;
    update_expression* cuda_allocate(allocation_helper* helper) override;
    unsigned int get_depth() const override;
};

#endif

