#pragma once

#ifndef UPDATE_EXPRESSION
#define UPDATE_EXPRESSION

#include "../common.h"
#include "cuda_stack.h"

#define NO_V_ID (-1U)
#define NO_VALUE (0)

enum expression_type
{
    literal_e = 0,
    clock_variable_e,
    system_variable_e,
    plus_e,
    minus_e,
    multiply_e,
    division_e
};


class update_expression final
{
private:
    expression_type type_;
    update_expression* left_;
    update_expression* right_;
    int value_;
    unsigned int variable_id_;
    //unsigned int operate(expression_type type, unsigned int left, unsigned int right);
    explicit  update_expression(expression_type type,
        update_expression* left = nullptr,
        update_expression* right = nullptr,
        int value = 0,
        unsigned int variable_id = NO_V_ID
        );
public:

    //SIMULATION methods
    GPU CPU update_expression* get_left() const;
    GPU CPU update_expression* get_right() const;
    GPU CPU void evaluate(cuda_stack<int>* stack,
        const lend_array<clock_timer_t>* timers, const lend_array<system_variable>* variables) const;
    

    //HOST methods
    std::string type_to_string();
    std::string to_string();
    int get_value();
    void accept(visitor* v) const;
    unsigned int get_depth() const;
    void cuda_allocate(update_expression* cuda_p, const allocation_helper* helper) const;

    //FACTORY CONSTRUCTORS
    static update_expression* literal_expression(int value);
    static update_expression* clock_expression(unsigned int clock_id);
    static update_expression* variable_expression(unsigned int variable);
    
    static update_expression* plus_expression(update_expression* left, update_expression* right);
    static update_expression* minus_expression(update_expression* left, update_expression* right);
    static update_expression* multiply_expression(update_expression* left, update_expression* right);
    static update_expression* division_expression(update_expression* left, update_expression* right);
};

#endif