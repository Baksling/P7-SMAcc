#pragma once

#ifndef UPDATE_EXPRESSION
#define UPDATE_EXPRESSION

#include "../common.h"
#include "cuda_stack.h"

#define NO_V_ID (-1U)
#define NO_VALUE (0)

enum expression_type
{
    //value types
    literal_e = 0,
    clock_variable_e,
    system_variable_e,

    //arithmatic types
    plus_e,
    minus_e,
    multiply_e,
    division_e,
    power_e,
    negation_e,
    sqrt_e,

    //boolean types
    less_equal_e,
    greater_equal_e,
    less_e,
    greater_e,
    equal_e,
    not_equal_e,
    not_e,

    //conditional types
    conditional_e
};


class update_expression final
{
private:
    expression_type type_;
    update_expression* condition_;
    update_expression* left_;
    update_expression* right_;
    double value_;
    unsigned int variable_id_;
    //unsigned int operate(expression_type type, unsigned int left, unsigned int right);
    explicit  update_expression(expression_type type,
        double value = 0,
        unsigned int variable_id = NO_V_ID,
        update_expression* left = nullptr,
        update_expression* right = nullptr,
        update_expression* condition = nullptr
        );
public:

    //SIMULATION methods
    GPU CPU update_expression* get_left() const;
    GPU CPU update_expression* get_right(const cuda_stack<double>* value_stack) const;
    GPU CPU void evaluate(cuda_stack<double>* stack,
                          const lend_array<clock_timer_t>* timers, const lend_array<system_variable>* variables) const;
    GPU CPU bool is_leaf() const;
    

    //HOST methods
    std::string type_to_string();
    std::string to_string();
    int get_value();
    void accept(visitor* v) const;
    unsigned int get_depth() const;
    void cuda_allocate(update_expression* cuda_p, const allocation_helper* helper) const;

    //FACTORY CONSTRUCTORS
    static update_expression* literal_expression(double value);
    static update_expression* clock_expression(unsigned int clock_id);
    static update_expression* variable_expression(unsigned int variable_id);
    
    static update_expression* plus_expression(update_expression* left, update_expression* right);
    static update_expression* minus_expression(update_expression* left, update_expression* right);
    static update_expression* multiply_expression(update_expression* left, update_expression* right);
    static update_expression* division_expression(update_expression* left, update_expression* right);
    
    static update_expression* power_expression(update_expression* left, update_expression* right);
    static update_expression* negate_expression(update_expression* expression);
    static update_expression* sqrt_expression(update_expression* expression);
    static update_expression* less_equal_expression(update_expression* left, update_expression* right);
    static update_expression* less_expression(update_expression* left, update_expression* right);
    static update_expression* greater_equal_expression(update_expression* left, update_expression* right);
    static update_expression* greater_expression(update_expression* left, update_expression* right);
    static update_expression* equal_expression(update_expression* left, update_expression* right);
    static update_expression* not_equal_expression(update_expression* left, update_expression* right);
    static update_expression* not_expression(update_expression* expression);

    static update_expression* conditional_expression(update_expression* condition,
                                                     update_expression* left, update_expression* right);
};

#endif