﻿#pragma once

#define NO_ID (-1)
#define UNUSED_VALUE (-1.0f)
#define BIG_DOUBLE (9999999.99)
#define UNLIMITED_TIME (-2.0)

#include "common.h"
#include "timer_t.h"


enum logical_operator
{
    less_equal = 0,
    greater_equal,
    less,
    greater,
    equal,
    not_equal,
    And,
    Or,
    Not,
};

class constraint_t : public element
{
private:
    constraint_t* con1_ = nullptr;
    constraint_t* con2_ = nullptr;
    int timer_id1_ = NO_ID;
    int timer_id2_ = NO_ID;
    float value_;
    logical_operator type_;
    bool get_bool_value(const constraint_t* con, const lend_array<timer_t>* timer_arr) const;
    double get_logical_value(int timer_id, const lend_array<timer_t>* timer_arr) const;
    bool validate_type() const;

    explicit constraint_t(logical_operator type, constraint_t* con1 = nullptr, constraint_t* con2 = nullptr,
                          int timer_id1 = NO_ID, int timer_id2 = NO_ID, float value = UNUSED_VALUE);
public:

    bool evaluate(const lend_array<timer_t>* timer_arr) const;
    void find_children(std::list<constraint_t*>* child_lst);
    logical_operator get_type() const;
    double max_time_progression(const lend_array<timer_t>* timers, double max_progression = 100.0);
    void accept(visistor& v) override;
    int get_timer1_id() const;
    int get_timer2_id() const;
    float get_value() const;

    //FACTORY CONSTRUCTORS
    inline static constraint_t less_equal_v(int timer_id, float value);
    inline static constraint_t less_equal_t(int timer_id, int timer_id2);
    
    inline static constraint_t greater_equal_v(int timer_id, float value);
    inline static constraint_t greater_equal_t(int timer_id, int timer_id2);
    
    inline static constraint_t less_v(int timer_id, float value);
    inline static constraint_t less_t(int timer_id, int timer_id2);
    
    inline static constraint_t greater_v(int timer_id, float value);
    inline static constraint_t greater_t(int timer_id, int timer_id2);
    
    inline static constraint_t equal_v(int timer_id, float value);
    inline static constraint_t equal_t(int timer_id, int timer_id2);

    inline static constraint_t not_equal_v(int timer_id, float value);
    inline static constraint_t not_equal_t(int timer_id, int timer_id2);


    //----------------|boolean constraints|----------------
    
    inline static constraint_t not_constraint(constraint_t* constraint);

    inline static constraint_t or_constraint(constraint_t* constraint1, constraint_t* constraint2);
    
    inline static constraint_t and_constraint(constraint_t* constraint1, constraint_t* constraint2);
};