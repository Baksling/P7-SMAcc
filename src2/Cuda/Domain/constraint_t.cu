#include "constraint_t.h"


GPU CPU double cuda_abs(const double f)
{
    return f < 0 ? -f : f;
}


constraint_t::constraint_t(const logical_operator_t type, const int timer_id1, const int timer_id2, const float value)
{
    this->type_ = type;
    this->timer_id1_ = timer_id1;
    this->timer_id2_ = timer_id2;
    this->value_ = value;
}

CPU GPU bool constraint_t::evaluate(const lend_array<clock_variable>* timers) const
{
    const double v1 = timers->at(this->timer_id1_)->get_temp_time();
    const double v2 = this->timer_id2_ == NO_ID
                ? static_cast<double>(this->value_)
                : timers->at(this->timer_id2_)->get_temp_time();
    
    switch(this->type_)
    {
        case logical_operator_t::less_equal_t: return v1 <= v2;
        case logical_operator_t::greater_equal_t: return v1 >= v2;
        case logical_operator_t::less_t: return v1 < v2;
        case logical_operator_t::greater_t: return v1 > v2;
        case logical_operator_t::equal_t: return cuda_abs(v1 - v2) < 0.005; //v1 == v2;
        case logical_operator_t::not_equal_t: return cuda_abs(v1 - v2) >= 0.005; //v1 != v2;
    }
    return false;
}

CPU GPU bool constraint_t::check_max_time_progression(const lend_array<clock_variable>* timer_arr, double* out_max_progression) const
{
    if(this->timer_id2_ != NO_ID) return false;
    if(this->type_ == logical_operator_t::less_t || this->type_ == logical_operator_t::less_equal_t)
    {
        const double time = timer_arr->at(this->timer_id1_)->get_time();
        const double value = static_cast<double>(this->value_);

        double diff = value - time;
        if(diff < 0) diff = 0.0;
        (*out_max_progression) = diff;
        return true;
    }
    
    return false;
}

void constraint_t::accept(visitor* v)
{
    return;
}

void constraint_t::pretty_print() const
{
    printf("Constraint type: %s | Timer 1 id: %3d | Timer 2 id: %3d | value: %10f \n", constraint_t::logical_operator_to_string(this->type_).c_str(),
           this->timer_id1_, this->timer_id2_, this->value_);
}

void constraint_t::cuda_allocate(constraint_t** pointer, const allocation_helper* helper) const
{
    cudaMalloc(pointer, sizeof(constraint_t));
    helper->free_list->push_back(*pointer);
    const constraint_t con = constraint_t(this->type_, this->timer_id1_, this->timer_id2_, this->value_);
    cudaMemcpy(*pointer, &con, sizeof(constraint_t), cudaMemcpyHostToDevice);
}

void constraint_t::cuda_allocate_2(constraint_t* cuda_pointer, const allocation_helper* helper) const
{
    const constraint_t con = constraint_t(this->type_, this->timer_id1_, this->timer_id2_, this->value_);
    cudaMemcpy(cuda_pointer, &con, sizeof(constraint_t), cudaMemcpyHostToDevice);
}

std::string constraint_t::logical_operator_to_string(const logical_operator_t op)
{
    switch (op)
    {
    case logical_operator_t::less_equal_t:
        return "<=";
    case logical_operator_t::greater_equal_t:
        return ">=";
    case logical_operator_t::less_t:
        return "<";
    case logical_operator_t::greater_t:
        return ">";
    case logical_operator_t::equal_t:
        return "==";
    case logical_operator_t::not_equal_t:
        return "!=";
    default:
        return "not a boolean operator";
    }
}

//! LESS THAN OR EQUAL
constraint_t* constraint_t::less_equal_v(const int timer_id, const float value)
{
    return new constraint_t{logical_operator_t::less_equal_t, timer_id, NO_ID, value};
}

constraint_t* constraint_t::less_equal_t(const int timer_id, const int timer_id2)
{
    return new constraint_t{logical_operator_t::less_equal_t, timer_id, timer_id2, UNUSED_VALUE};

}

//! GREATER THAN OR EQUAL
constraint_t* constraint_t::greater_equal_v(const int timer_id, const float value)
{
    return new constraint_t{logical_operator_t::greater_equal_t, timer_id, NO_ID, value};

}

constraint_t* constraint_t::greater_equal_t(const int timer_id, const int timer_id2)
{
    return new constraint_t{logical_operator_t::greater_equal_t, timer_id, timer_id2, UNUSED_VALUE};
}

//! LESS THAN
constraint_t* constraint_t::less_v(const int timer_id, const float value)
{
    return new constraint_t{logical_operator_t::less_t, timer_id, NO_ID, value};

}

constraint_t* constraint_t::less_t(const int timer_id, const int timer_id2)
{
    return new constraint_t{logical_operator_t::less_t, timer_id, timer_id2, UNUSED_VALUE};
}

//! GREATER THAN
constraint_t* constraint_t::greater_v(const int timer_id, const float value)
{
    return new constraint_t{logical_operator_t::greater_t, timer_id, NO_ID, value};

}

constraint_t* constraint_t::greater_t(const int timer_id, const int timer_id2)
{
    return new constraint_t{logical_operator_t::greater_t, timer_id, timer_id2, UNUSED_VALUE};
}

//! equal
constraint_t* constraint_t::equal_v(const int timer_id, const float value)
{
    return new constraint_t{logical_operator_t::equal_t, timer_id, NO_ID, value};

}

constraint_t* constraint_t::equal_t(const int timer_id, const int timer_id2)
{
    return new constraint_t{logical_operator_t::equal_t, timer_id, timer_id2, UNUSED_VALUE};
}

//! NOT EQUAL
constraint_t* constraint_t::not_equal_v(const int timer_id, const float value)
{
    return new constraint_t{logical_operator_t::not_equal_t, timer_id, NO_ID, value};
}

constraint_t* constraint_t::not_equal_t(const int timer_id, const int timer_id2)
{
    return new constraint_t{logical_operator_t::not_equal_t, timer_id, timer_id2, UNUSED_VALUE};
}
