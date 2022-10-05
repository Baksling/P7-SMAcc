#include "constraint_tt.h"


GPU CPU double cuda_abs(const double f)
{
    return f < 0 ? -f : f;
}


constraint_tt::constraint_tt(logical_operator_t type, int timer_id1, int timer_id2, float value)
{
    this->type_ = type;
    this->timer_id1_ = timer_id1;
    this->timer_id2_ = timer_id2;
    this->value_ = value;
}

logical_operator_t constraint_tt::get_type() const
{
    return this->type_;
}

bool constraint_tt::evaluate(const lend_array<clock_timer_t>* timers) const
{
    const double v1 = timers->at(this->timer_id1_)->get_time();
    const double v2 = this->timer_id2_ == NO_ID
                ? static_cast<double>(this->value_)
                : timers->at(this->timer_id2_)->get_time();

    switch(this->type_)
    {
        case less_equal_t: return v1 <= v2;
        case greater_equal_t: return v1 >= v2;
        case less_t: return v1 < v2;
        case greater_t: return v1 > v2;
        case equal_t: return cuda_abs(v1 - v2) < 0.01; //v1 == v2;
        case not_equal_t: return cuda_abs(v1 - v2) >= 0.01; //v1 != v2;
    }
    return false;
}

double constraint_tt::max_time_progression(const lend_array<clock_timer_t>* timer_arr, double max_progression) const
{
    if(this->timer_id1_ != NO_ID && this->timer_id2_ != NO_ID)
        return BIG_DOUBLE;

    if(this->type_ == logical_operator_t::less_t || this->type_ == logical_operator_t::less_equal_t)
    {
        const double time = timer_arr->at(this->timer_id1_)->get_time();
        const double value = static_cast<double>(this->value_);

        double diff = value - time;
        if(diff < 0) diff = 0.0;
        max_progression = diff >= 0 && diff < max_progression ? diff : max_progression;
        return max_progression;
    }
    
    return BIG_DOUBLE;
}

void constraint_tt::accept(visitor* v)
{
    return;
}

int constraint_tt::get_timer1_id() const
{
    return this->timer_id1_;
}

int constraint_tt::get_timer2_id() const
{
    return this->timer_id2_;
}

float constraint_tt::get_value() const
{
    return this->value_;
}
