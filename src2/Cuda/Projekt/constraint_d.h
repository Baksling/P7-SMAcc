#pragma once
#include <limits>

#include "common.h"
#include "timer_d.h"
#include "uneven_list.h"

#define NO_ID (-1)
#define UNUSED_VALUE (-1.0f)
#define BIG_DOUBLE (9999999.99)
#define UNLIMITED_TIME (-2.0)


enum logical_operator2
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

class constraint_d
{
private:
    int sid1_;
    int sid2_;
    float value_;
    logical_operator2 type_;
    GPU CPU bool get_bool_value(const int constraint_id, const array_info<timer_d>* timer_arr, const array_info<constraint_d>* constraint_arr) const;
    GPU CPU double get_logical_value(const int timer_id, const array_info<timer_d>* timer_arr) const;
    constraint_d(const int sid1, const int sid2, const float value, const logical_operator2 type)
    {
        this->sid1_ = sid1;
        this->sid2_ = sid2;
        this->value_ = value;
        this->type_ = type;
    }
public:

    GPU CPU bool evaluate(const array_info<timer_d>* timer_arr, const array_info<constraint_d>* constraint_arr) const;
    GPU CPU void find_children(list<constraint_d*>* child_lst, array_info<constraint_d>* all_constraints);
    GPU CPU logical_operator2 get_type() const;
    GPU CPU double get_difference(const array_info<timer_d>* timers) const;
    
    //FACTORY CONSTRUCTORS
    inline static constraint_d less_equal_v(int timer_id, float value);
    inline static constraint_d less_equal_t(int timer_id, int timer_id2);
    
    inline static constraint_d greater_equal_v(int timer_id, float value);
    inline static constraint_d greater_equal_t(int timer_id, int timer_id2);
    
    inline static constraint_d less_v(int timer_id, float value);
    inline static constraint_d less_t(int timer_id, int timer_id2);
    
    inline static constraint_d greater_v(int timer_id, float value);
    inline static constraint_d greater_t(int timer_id, int timer_id2);
    
    inline static constraint_d equal_v(int timer_id, float value);
    inline static constraint_d equal_t(int timer_id, int timer_id2);

    inline static constraint_d not_equal_v(int timer_id, float value);
    inline static constraint_d not_equal_t(int timer_id, int timer_id2);


    //----------------|boolean constraints|----------------
    
    inline static constraint_d not_constraint(int constraint_id);

    inline static constraint_d or_constraint(int constraint_id1, int constraint_id2);
    
    inline static constraint_d and_constraint(int constraint_id1, int constraint_id2);
};

