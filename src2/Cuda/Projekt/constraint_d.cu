#include "constraint_d.h"


GPU CPU double cuda_abs(const double f)
{
    return f < 0 ? -f : f;
}

GPU CPU bool is_boolean_operator(const logical_operator2 op)
{
    switch (op)
    {
    case logical_operator2::less_equal: return false;
    case logical_operator2::greater_equal: return false;
    case logical_operator2::less: return false;
    case logical_operator2::greater: return false;
    case logical_operator2::equal: return false;
    case not_equal: return false;
    case And: return true;
    case Or: return true;
    case Not: return true;
    }
    return false;
}

GPU CPU bool constraint_d::get_bool_value(const int constraint_id,
    const array_info<timer_d>* timer_arr, const array_info<constraint_d>* constraint_arr) const
{
    if(!is_boolean_operator(this->type_)) return false;
    if (constraint_id == NO_ID) return false;
    return constraint_arr->arr[constraint_id].evaluate(timer_arr, constraint_arr);
}

CPU GPU double constraint_d::get_logical_value(const int timer_id, const array_info<timer_d>* timer_arr) const
{
    if(is_boolean_operator(this->type_)) return UNUSED_VALUE;
    const bool has_id = timer_id != NO_ID;
    const bool has_value = this->value_ >= UNUSED_VALUE;
    if(!has_id && !has_value)
    {
        printf("Constraint contains neither timer nor value. PANIC!");
    }
    
    return has_id ? timer_arr->arr[timer_id].get_value() : static_cast<double>(this->value_);
}

GPU CPU bool constraint_d::evaluate(const array_info<timer_d>* timer_arr, const array_info<constraint_d>* constraint_arr) const
{
    const bool  b1 = this->get_bool_value(this->sid1_, timer_arr, constraint_arr);
    const bool  b2 = this->get_bool_value(this->sid2_, timer_arr, constraint_arr);
    const double v1 = this->get_logical_value(this->sid1_, timer_arr);
    const double v2 = this->get_logical_value(this->sid2_, timer_arr);
    
    switch (this->type_) {
        case logical_operator2::less_equal: return v1 <= v2;
        case logical_operator2::greater_equal: return v1 >= v2;
        case logical_operator2::less: return v1 < v2;
        case logical_operator2::greater: return v1 > v2;
        case logical_operator2::equal: return cuda_abs(v1 - v2) < 0.01; //v1 == v2;
        case logical_operator2::not_equal: return abs(v1 - v2) >= 0.01; //v1 != v2;
        case And: return b1 && b2;
        case Or: return b1 || b2;
        case Not: return !b1;
    }
    return false;
}

void constraint_d::find_children(list<constraint_d*>* child_lst, array_info<constraint_d>* all_constraints)
{
    switch (this->type_)
    {
    case logical_operator2::less_equal: break;
    case logical_operator2::greater_equal: break;
    case logical_operator2::less: break;
    case logical_operator2::greater: break;
    case logical_operator2::equal: break;
    case logical_operator2::not_equal: break;
    case And: case Or:
        all_constraints->arr[this->sid1_].find_children(child_lst, all_constraints);
        all_constraints->arr[this->sid2_].find_children(child_lst, all_constraints);
        break;
    case Not:
        all_constraints->arr[this->sid1_].find_children(child_lst, all_constraints);
    }

    child_lst->push_back(this);

}

logical_operator2 constraint_d::get_type() const
{
    return this->type_;
}

double constraint_d::get_difference(const array_info<timer_d>* timers) const
{
    if(is_boolean_operator(this->type_)) return BIG_DOUBLE;
    if(!(this->type_ == logical_operator2::less_equal || this->type_ == logical_operator2::less)) return BIG_DOUBLE;

    const double t1 = timers->arr[this->sid1_].get_value();
    if(this->value_ > UNUSED_VALUE)
    {
        return t1 - static_cast<double>(this->value_);
    }

    //TODO difference between two timers not supported atm
    return UNLIMITED_TIME;
    const double t2 = timers->arr[this->sid2_].get_value();
    return t1 <= t2 ? UNLIMITED_TIME : BIG_DOUBLE;
}


//! LESS THAN OR EQUAL
constraint_d constraint_d::less_equal_v(const int timer_id, const float value)
{
    return constraint_d{timer_id, NO_ID, value, logical_operator2::less_equal};
}

constraint_d constraint_d::less_equal_t(const int timer_id, const int timer_id2)
{
    return constraint_d{timer_id, timer_id2, UNUSED_VALUE, logical_operator2::less_equal};

}

//! GREATER THAN OR EQUAL
constraint_d constraint_d::greater_equal_v(const int timer_id, const float value)
{
    return constraint_d{timer_id, NO_ID, value, logical_operator2::greater_equal};
}

constraint_d constraint_d::greater_equal_t(const int timer_id, const int timer_id2)
{
    return constraint_d{timer_id, timer_id2, UNUSED_VALUE, logical_operator2::greater_equal};
}

//! LESS THAN
constraint_d constraint_d::less_v(const int timer_id, const float value)
{
    return constraint_d{timer_id, NO_ID, value, logical_operator2::less};
}

constraint_d constraint_d::less_t(const int timer_id, const int timer_id2)
{
    return constraint_d{timer_id, timer_id2, UNUSED_VALUE, logical_operator2::less};
}

//! GREATER THAN
constraint_d constraint_d::greater_v(const int timer_id, const float value)
{
    return constraint_d{timer_id, NO_ID, value, logical_operator2::greater};
}

constraint_d constraint_d::greater_t(const int timer_id, const int timer_id2)
{
    return constraint_d{timer_id, timer_id2, UNUSED_VALUE, logical_operator2::greater};
}

//! equal
constraint_d constraint_d::equal_v(const int timer_id, const float value)
{
    return constraint_d{timer_id, NO_ID, value, logical_operator2::equal};
}

constraint_d constraint_d::equal_t(const int timer_id, const int timer_id2)
{
    return constraint_d{timer_id, timer_id2, UNUSED_VALUE, logical_operator2::equal};
}

//! NOT EQUAL
constraint_d constraint_d::not_equal_v(const int timer_id, const float value)
{
    return constraint_d{timer_id, NO_ID, value, logical_operator2::not_equal};
}

constraint_d constraint_d::not_equal_t(const int timer_id, const int timer_id2)
{
    return constraint_d{timer_id, timer_id2, UNUSED_VALUE, logical_operator2::not_equal};
}

constraint_d constraint_d::not_constraint(const int constraint_id)
{
    return constraint_d{constraint_id, NO_ID, UNUSED_VALUE, logical_operator2::Not};
}

constraint_d constraint_d::or_constraint(const int constraint_id1, const int constraint_id2)
{
    return constraint_d{constraint_id1, constraint_id2, UNUSED_VALUE, logical_operator2::not_equal};
}

constraint_d constraint_d::and_constraint(const int constraint_id1, const int constraint_id2)
{
    return constraint_d{constraint_id1, constraint_id2, UNUSED_VALUE, logical_operator2::And};
}


