#pragma once

#ifndef UPDATE_EXPRESSION
#define UPDATE_EXPRESSION

#include "../../common/macro.h"
#include "../../common/cuda_stack.h"
#include "../../Visitors/visitor.h"

#include <string>

#define NO_V_ID (-1U)
#define NO_VALUE (0)

// Prototype
class simulator_state;

enum expression_type
{
    //value types
    literal_e = 0,
    clock_variable_e,
    system_variable_e,

    //random
    random_e,

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


class expression final
{
private:
    expression_type type_;
    expression* left_;
    expression* right_;
    union
    {
        expression* condition;
        double value;
        unsigned int variable_id;  
    };
    
    //unsigned int operate(expression_type type, unsigned int left, unsigned int right);
    explicit  expression(expression_type type,
        double value = 0,
        unsigned int variable_id = NO_V_ID,
        expression* left = nullptr,
        expression* right = nullptr,
        expression* condition = nullptr
        );
public:

    //SIMULATION methods
    GPU CPU expression* get_left() const;
    GPU CPU expression* get_right(const cuda_stack<double>* value_stack) const;
    
    GPU CPU double evaluate_current(simulator_state* state) const;
    GPU CPU double evaluate(simulator_state* state);
    
    GPU CPU bool is_leaf() const;
    

    //HOST methods
    std::string type_to_string() const;
    std::string to_string() const;
    void pretty_print(std::ostream& os) const;
    void accept(visitor* v) const;
    unsigned int get_depth() const;
    bool is_constant() const;
    bool contains_clock_expression() const;
    void cuda_allocate(expression* cuda_p, allocation_helper* helper) const;

    //FACTORY CONSTRUCTORS
    static expression* literal_expression(double value);
    static expression* clock_expression(unsigned int clock_id);
    static expression* variable_expression(unsigned int variable_id);
    
    static expression* random_expression(double max);
    
    static expression* plus_expression(expression* left, expression* right);
    static expression* minus_expression(expression* left, expression* right);
    static expression* multiply_expression(expression* left, expression* right);
    static expression* division_expression(expression* left, expression* right);
    
    static expression* power_expression(expression* left, expression* right);
    static expression* negate_expression(expression* expr);
    static expression* sqrt_expression(expression* expr);
    static expression* less_equal_expression(expression* left, expression* right);
    static expression* less_expression(expression* left, expression* right);
    static expression* greater_equal_expression(expression* left, expression* right);
    static expression* greater_expression(expression* left, expression* right);
    static expression* equal_expression(expression* left, expression* right);
    static expression* not_equal_expression(expression* left, expression* right);
    static expression* not_expression(expression* expr);

    static expression* conditional_expression(expression* condition,
                                                     expression* left, expression* right);
};

#endif