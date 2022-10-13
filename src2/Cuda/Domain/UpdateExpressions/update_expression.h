#pragma once

#ifndef UPDATE_EXPRESSION_H
#define UPDATE_EXPRESSION_H

#include "cuda_stack.h"
#include "../common.h"


class update_expression
{
public:
    virtual ~update_expression() = default;
    update_expression() = default;

    virtual void evaluate(cuda_stack<double>* stack, lend_array<clock_timer_t>* timers, lend_array<system_variable>* variables);
    virtual void accept(visitor* v) = 0;
    virtual unsigned int get_depth() const = 0;
    virtual update_expression* cuda_allocate(allocation_helper* helper) = 0;
};


#endif