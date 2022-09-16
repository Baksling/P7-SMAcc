#include "Guard.h"
#include "Timer.h"

using namespace std;

guard::guard(const logical_operator type, const double value, timer* timer)
{
    this->type_ = type;
    this->value_ = value;
    this->timer_ = timer;
}

guard::guard(const logical_operator type, const double value, int timer_id)
{
    this->type_ = type;
    this->value_ = value;
}

double guard::get_value() const
{
    return this->value_;
}

bool guard::validate_guard()
{
    switch (this->type_)
    {
    case less_equal: return this->timer_->get_time() <= this->get_value();
    case greater_equal: return this->timer_->get_time() >= this->get_value();
    case less: return this->timer_->get_time() < this->get_value();
    case greater: return this->timer_->get_time() > this->get_value();
    case equal: return this->timer_->get_time() == this->get_value();
    case not_equal: return this->timer_->get_time() != this->get_value();
    }

    return false;
}

logical_operator guard::get_type()
{
    return this->type_;
}

