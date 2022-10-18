#pragma once

// #include "common.h"

#include "../common/macro.h"
#include "../common/lend_array.h"
#include "../common/allocation_helper.h"
#include "clock_variable.h"
#include "../Visitors/visitor.h"


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

    //SIMULATOR METHODS
    CPU GPU bool evaluate(const lend_array<clock_variable>* timers) const;
    CPU GPU bool check_max_time_progression(const lend_array<clock_variable>* timer_arr, double* out_max_progression) const;
    

    //HOST METHODS
    logical_operator_t get_type() const;
    int get_timer1_id() const;
    int get_timer2_id() const;
    float get_value() const;
    void accept(visitor* v);
    void cuda_allocate(constraint_t** pointer, const allocation_helper* helper) const;
    void cuda_allocate_2(constraint_t* cuda_pointer, const allocation_helper* helper) const;
    CPU static std::string to_string(logical_operator_t op);


    //FACTORY CONSTRUCTORS
    static constraint_t* less_equal_v(int timer_id, float value);
    static constraint_t* less_equal_t(int timer_id, int timer_id2);
    
    static constraint_t* greater_equal_v(int timer_id, float value);
    static constraint_t* greater_equal_t(int timer_id, int timer_id2);
    
    static constraint_t* less_v(int timer_id, float value);
    static constraint_t* less_t(int timer_id, int timer_id2);
    
    static constraint_t* greater_v(int timer_id, float value);
    static constraint_t* greater_t(int timer_id, int timer_id2);
    
    static constraint_t* equal_v(int timer_id, float value);
    static constraint_t* equal_t(int timer_id, int timer_id2);

    static constraint_t* not_equal_v(int timer_id, float value);
    static constraint_t* not_equal_t(int timer_id, int timer_id2);
};
