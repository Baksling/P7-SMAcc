#pragma once

#include "common.h"

#define NO_ID (-1)
#define UNUSED_VALUE (-1.0f)
#define UNUSED_VALUE (-1.0f)
#define BIG_DOUBLE (9999999.99)


enum logical_operator_t
{
    less_equal_t = 0,
    greater_equal_t,
    less_t,
    greater_t,
    equal_t,
    not_equal_t
};

class constraint_t
{
    int timer_id1_;
    int timer_id2_;
    float value_;
    logical_operator_t type_;
    explicit constraint_t(logical_operator_t type,
        int timer_id1 = NO_ID, int timer_id2 = NO_ID, float value = UNUSED_VALUE );
public:
    GPU CPU logical_operator_t get_type() const;
    GPU CPU bool evaluate(const lend_array<clock_timer_t>* timers) const;
    GPU double max_time_progression(const lend_array<clock_timer_t>* timer_arr, double max_progression = 100.0) const;
    void accept(visitor* v);
    void cuda_allocate(constraint_t** pointer, const allocation_helper* helper) const;
 
    
    CPU GPU int get_timer1_id() const;
    GPU CPU int get_timer2_id() const;
    CPU GPU float get_value() const;


    //FACTORY CONSTRUCTORS
    static constraint_t less_equal_v(int timer_id, float value);
    static constraint_t less_equal_t(int timer_id, int timer_id2);
    
    static constraint_t greater_equal_v(int timer_id, float value);
    static constraint_t greater_equal_t(int timer_id, int timer_id2);
    
    static constraint_t less_v(int timer_id, float value);
    static constraint_t less_t(int timer_id, int timer_id2);
    
    static constraint_t greater_v(int timer_id, float value);
    static constraint_t greater_t(int timer_id, int timer_id2);
    
    static constraint_t equal_v(int timer_id, float value);
    static constraint_t equal_t(int timer_id, int timer_id2);

    static constraint_t not_equal_v(int timer_id, float value);
    static constraint_t not_equal_t(int timer_id, int timer_id2);
};
