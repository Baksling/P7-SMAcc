#include "constraint_t.h"

GPU CPU double cuda_abs(const double f)
{
    return f < 0 ? -f : f;
}

GPU CPU bool is_boolean_operator(const logical_operator op)
{
    switch (op)
    {
    case less_equal: return false;
    case greater_equal: return false;
    case less: return false;
    case greater: return false;
    case equal: return false;
    case not_equal: return false;
    case And: return true;
    case Or: return true;
    case Not: return true;
    }
    return false;
}

GPU bool constraint_t::get_bool_value(const constraint_t* con, const lend_array<clock_timer_t>* timer_arr) const
{
    if(!is_boolean_operator(this->type_)) return false;
    if (con == nullptr) return false;
    return con->evaluate(timer_arr);
}

GPU double constraint_t::get_logical_value(const int timer_id, const lend_array<clock_timer_t>* timer_arr) const
{
    if(is_boolean_operator(this->type_)) return BIG_DOUBLE;


    //! ASSUMED TO BE HANDLED BY constraint_t::validate_type
    // if(timer_id == NO_ID && this->value_ < 0)
    // {
    //     printf("a logical constraint with neither value nor timer_id evaluated");
    // }
    
    return timer_id == NO_ID
    ? static_cast<double>(this->value_)
    : timer_arr->at(timer_id)->get_time();
}

GPU CPU bool constraint_t::validate_type() const
{
    switch (this->type_)
    {
    case less_equal:
    case greater_equal:
    case less: 
    case greater:
    case equal:
    case not_equal:
        return this->timer_id1_ != NO_ID && (this->value_ >= 0 || timer_id2_ != NO_ID);
    case And:
    case Or:
        return this->con1_ != nullptr && this->con2_ != nullptr;
    case Not:
        return this->con1_ != nullptr;
    }
    return false;
}

constraint_t::constraint_t(const logical_operator type, constraint_t* con1, constraint_t* con2, const int timer_id1,
                           const int timer_id2, const float value)
{
    this->type_ = type;
    this->con1_ = con1;
    this->con2_ = con2;
    this->value_ = value;
    this->timer_id1_ = timer_id1;
    this->timer_id2_ = timer_id2;
    if(!this->validate_type())
    {
        throw std::invalid_argument( "The constraint is invalid >:(" );
    }
}

GPU bool constraint_t::evaluate(const lend_array<clock_timer_t>* timer_arr) const
{
    const bool  b1  = this->get_bool_value(   this->con1_,      timer_arr);
    const bool  b2  = this->get_bool_value(   this->con2_,      timer_arr);
    const double v1 = this->get_logical_value(this->timer_id1_, timer_arr);
    const double v2 = this->get_logical_value(this->timer_id2_, timer_arr);
    
    switch (this->type_) {
    case less_equal: return v1 <= v2;
    case greater_equal: return v1 >= v2;
    case less: return v1 < v2;
    case greater: return v1 > v2;
    case equal: return cuda_abs(v1 - v2) < 0.01; //v1 == v2;
    case not_equal: return cuda_abs(v1 - v2) >= 0.01; //v1 != v2;
    case And: return b1 && b2;
    case Or: return b1 || b2;
    case Not: return !b1;
    }
    return false;
}

// GPU CPU void constraint_t::find_children(std::list<constraint_t*>* child_lst)
// {
//     // if (this->con1_ != nullptr)
//     // {
//     //     this->con1_->find_children(child_lst);
//     // }
//     // if(this->con2_ != nullptr)
//     // {
//     //     this->con2_->find_children(child_lst);
//     // }
//     // child_lst->push_back(this);
// }

GPU CPU logical_operator constraint_t::get_type() const
{
    return this->type_;
}

GPU double constraint_t::max_time_progression(const lend_array<clock_timer_t>* timers, double max_progression)
{
    // std::list<constraint_t*> constraint_lst;
    // this->find_children(&constraint_lst);
    //
    // if(max_progression < 0.0) max_progression = 0.0;
    //
    // for(const constraint_t* con : constraint_lst)
    // {
    //     const logical_operator type = con->get_type();
    //     //only relevant if it is upper bounded logical operator.
    //     if(!(type == less_equal || type == less)) continue;
    //
    //     const double t1 = timers->at(this->timer_id1_)->get_time();
    //     //case that constraint is between timer and value
    //     if(this->value_ >= 0)
    //     {
    //         const double diff = static_cast<double>(this->value_) - t1;
    //         max_progression = diff < max_progression && diff >= 0 ? diff : max_progression;
    //     }
    // }

    return max_progression;
}

void constraint_t::accept(visistor& v)
{
    switch (this->type_)
    {
    case And:
    case Or:
        v.visit(this->con1_);
        v.visit(this->con2_);
        break;
    case Not:
        v.visit(this->con1_);
        break;
    default:
        break;
    }
}

int constraint_t::get_timer1_id() const
{
    return this->timer_id1_;
}
int constraint_t::get_timer2_id() const
{
    return this->timer_id2_;
}

float constraint_t::get_value() const
{
    return this->value_;
}


//! LESS THAN OR EQUAL
constraint_t constraint_t::less_equal_v(const int timer_id, const float value)
{
    return constraint_t{less_equal, nullptr, nullptr, timer_id, NO_ID, value};
}

constraint_t constraint_t::less_equal_t(const int timer_id, const int timer_id2)
{
    return constraint_t{less_equal, nullptr, nullptr, timer_id, timer_id2, UNUSED_VALUE};

}

//! GREATER THAN OR EQUAL
constraint_t constraint_t::greater_equal_v(const int timer_id, const float value)
{
    return constraint_t{greater_equal, nullptr, nullptr, timer_id, NO_ID, value};

}

constraint_t constraint_t::greater_equal_t(const int timer_id, const int timer_id2)
{
    return constraint_t{greater_equal, nullptr, nullptr, timer_id, timer_id2, UNUSED_VALUE};
}

//! LESS THAN
constraint_t constraint_t::less_v(const int timer_id, const float value)
{
    return constraint_t{less, nullptr, nullptr, timer_id, NO_ID, value};

}

constraint_t constraint_t::less_t(const int timer_id, const int timer_id2)
{
    return constraint_t{less, nullptr, nullptr, timer_id, timer_id2, UNUSED_VALUE};
}

//! GREATER THAN
constraint_t constraint_t::greater_v(const int timer_id, const float value)
{
    return constraint_t{greater, nullptr, nullptr, timer_id, NO_ID, value};

}

constraint_t constraint_t::greater_t(const int timer_id, const int timer_id2)
{
    return constraint_t{greater, nullptr, nullptr, timer_id, timer_id2, UNUSED_VALUE};
}

//! equal
constraint_t constraint_t::equal_v(const int timer_id, const float value)
{
    return constraint_t{equal, nullptr, nullptr, timer_id, NO_ID, value};

}

constraint_t constraint_t::equal_t(const int timer_id, const int timer_id2)
{
    return constraint_t{equal, nullptr, nullptr, timer_id, timer_id2, UNUSED_VALUE};
}

//! NOT EQUAL
constraint_t constraint_t::not_equal_v(const int timer_id, const float value)
{
    return constraint_t{not_equal, nullptr, nullptr, timer_id, NO_ID, value};
}

constraint_t constraint_t::not_equal_t(const int timer_id, const int timer_id2)
{
    return constraint_t{not_equal, nullptr, nullptr, timer_id, timer_id2, UNUSED_VALUE};
}

//NOT
constraint_t constraint_t::not_constraint(constraint_t* constraint)
{
    return constraint_t{Not, constraint, nullptr, NO_ID, NO_ID, UNUSED_VALUE};
}

//OR
constraint_t constraint_t::or_constraint(constraint_t* constraint1, constraint_t* constraint2)
{
    return constraint_t{Or, constraint1, constraint2, NO_ID, NO_ID, UNUSED_VALUE};
}

//AND
constraint_t constraint_t::and_constraint(constraint_t* constraint1, constraint_t* constraint2)
{
    return constraint_t{Not, constraint1, constraint2, NO_ID, NO_ID, UNUSED_VALUE};
}

