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
    CPU GPU double evaluate_expression(simulator_state* state) const;
    CPU GPU void apply_update(simulator_state* state) const;
    CPU GPU void apply_temp_update(simulator_state* state) const;
    CPU GPU void reset_temp_update(const simulator_state* state) const;
    
    //HOST METHODS
    int get_id() const;
    CPU GPU int get_timer_id() const;
    void accept(visitor* v) const;
    void cuda_allocate(update_t* cuda, const allocation_helper* helper) const;
    update_expression* get_expression_root() const; //haha bak is gonna get mad at me >:)
    static int get_expression_depth(const update_expression* exp);
};

#endif