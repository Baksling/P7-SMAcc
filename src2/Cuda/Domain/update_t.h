#pragma once

#ifndef UPDATE_T_H
#define UPDATE_T_H

#include "common.h"
#include "UpdateExpressions/update_expression.h"
class update_expression;
template<typename  T> class cuda_stack;

class update_t
{
private:
    int id_;
    int variable_id_;
    bool is_clock_update_;
    update_expression* expression_;
    cuda_stack<int>* value_stack_;
    cuda_stack<update_expression*>* expression_stack_;
    explicit update_t(const update_t* source, update_expression* expression,
        cuda_stack<int>* value_stack, cuda_stack<update_expression*>* evaluation_stack);
    
public:
    explicit update_t(int id, int variable_id, bool is_clock_update, update_expression* expression);

    //SIMULATOR METHODS
    CPU GPU int evaluate_expression(const lend_array<clock_timer_t>* timers, const lend_array<system_variable>* variables) const;
    CPU GPU void apply_update(
        const lend_array<clock_timer_t>* timers, const lend_array<system_variable>* variables) const;
    
    //HOST METHODS
    int get_id() const;
    CPU GPU int get_timer_id() const;
    void accept(visitor* v) const;
    void cuda_allocate(update_t* cuda, const allocation_helper* helper) const;
    update_expression* get_expression_root() const; //haha bak is gonna get mad at me >:)
    static int get_expression_depth(const update_expression* exp);
};

#endif