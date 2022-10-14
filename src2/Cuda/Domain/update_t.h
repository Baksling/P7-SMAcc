#pragma once

#ifndef UPDATE_T_H
#define UPDATE_T_H

#include "common.h"

class update_t
{
private:
    int id_;
    int variable_id_;
    bool is_clock_update_;
    update_expression* expression_;
    explicit update_t(const update_t* source, update_expression* expression);
    
public:
    explicit update_t(int id, int variable_id, bool is_clock_update, update_expression* expression);

    //SIMULATOR METHODS
    CPU GPU double evaluate_expression(
        cuda_stack<update_expression*>* expression_stack,
        cuda_stack<double>* value_stack,
        const lend_array<clock_timer_t>* timers,
        const lend_array<system_variable>* variables) const;
    CPU GPU void apply_update(cuda_stack<update_expression*>* expression_stack, cuda_stack<double>* value_stack, 
        const lend_array<clock_timer_t>* timers, const lend_array<system_variable>* variables) const;
    
    //HOST METHODS
    int get_id() const;
    CPU GPU int get_timer_id() const;
    void accept(visitor* v) const;
    void cuda_allocate(update_t* cuda, const allocation_helper* helper) const;
};

#endif