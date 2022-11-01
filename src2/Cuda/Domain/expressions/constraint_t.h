#pragma once

// #include "common.h"

#include "../../common/macro.h"
#include "../../common/allocation_helper.h"
#include "../../Visitors/visitor.h"
#include "../expressions/expression.h"

#define NO_ID (-1)
#define UNUSED_VALUE (-1.0f)
#define UNUSED_VALUE (-1.0f)
#define BIG_DOUBLE (9999999.99)

class simulator_state;

enum logical_operator_t
{
    less_equal_t = 0,
    greater_equal_t,
    less_t,
    greater_t,
    equal_t,
    not_equal_t
};



struct constraint_value
{
    constraint_value()
    {
        this->is_clock = true;
        this->expr = nullptr;
        this->clock_id = NO_ID;
    }
    
    
public:
    bool is_clock;
    union
    {
        int clock_id;
        expression* expr;
    };

    static constraint_value from_timer(const int clock_id)
    {
        constraint_value con = {};
        con.is_clock = true;
        con.clock_id = clock_id;
        return con;
    }
    
    static constraint_value from_expression(expression* expr)
    {
        constraint_value con = {};
        con.is_clock = false;
        con.expr = expr;
        return con;
    }
};

class constraint_t
{
    logical_operator_t type_;
    constraint_value left_;
    constraint_value right_;

    explicit constraint_t(logical_operator_t type, constraint_value left, constraint_value right, bool validate = true);
public:

    //SIMULATOR METHODS
    CPU GPU bool evaluate(simulator_state* state) const;
    CPU GPU bool check_max_time_progression(simulator_state* state, double* out_max_progression) const;
    

    //HOST METHODS
    void accept(visitor* v);
    void pretty_print() const;
    void cuda_allocate(constraint_t** pointer, const allocation_helper* helper) const;
    CPU static std::string logical_operator_to_string(logical_operator_t op);


    //FACTORY CONSTRUCTORS
    static constraint_t* less_equal_v(const int timer_id, expression* value_expr);
    static constraint_t* less_equal_e(expression* value_expr1, expression* value_expr2);
    static constraint_t* less_equal_t(int timer_id, int timer_id2);
    
    static constraint_t* greater_equal_v(const int timer_id, expression* value_expr);
    static constraint_t* greater_equal_e(expression* value_expr1, expression* value_expr2);
    static constraint_t* greater_equal_t(int timer_id, int timer_id2);
    
    static constraint_t* less_v(const int timer_id, expression* value_expr);
    static constraint_t* less_e(expression* value_expr1, expression* value_expr2);
    static constraint_t* less_t(int timer_id, int timer_id2);
    
    static constraint_t* greater_v(const int timer_id, expression* value_expr);
    static constraint_t* greater_e(expression* value_expr1, expression* value_expr2);
    static constraint_t* greater_t(int timer_id, int timer_id2);
    
    static constraint_t* equal_v(const int timer_id, expression* value_expr);
    static constraint_t* equal_e(expression* value_expr1, expression* value_expr2);
    static constraint_t* equal_t(int timer_id, int timer_id2);

    static constraint_t* not_equal_v(const int timer_id, expression* value_expr);
    static constraint_t* not_equal_e(expression* value_expr1, expression* value_expr2);
    static constraint_t* not_equal_t(int timer_id, int timer_id2);
};
