#pragma once

#ifndef UPDATE_T_H
#define UPDATE_T_H

#include "common.h"
#include "system_variable.h"
#include "UpdateExpressions/update_expression.h"

class update_t
{
private:
    int id_;
    int timer_id_;
    bool clock_update_;
    update_expression* expression_;
    cuda_stack<double> value_stack_{0};
    cuda_stack<update_expression*> evaluation_stack_{0};
    double evaluate_expression();
    
public:
    update_t(int id, int timer_id, bool is_clock_update, update_expression* expression);

    //SIMULATOR METHODS
    CPU GPU void apply_update(
        const lend_array<clock_timer_t>* timers, const lend_array<system_variable>* variables) const;
    
    //HOST METHODS
    int get_id() const;
    CPU GPU int get_timer_id() const;
    CPU GPU float get_timer_value() const;
    void accept(visitor* v);
    void cuda_allocate(update_t** pointer, const allocation_helper* helper) const;
};

#endif