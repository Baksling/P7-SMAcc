#include "timer_t.h"

CPU GPU timer_t::timer_t(const int id, const double start_value)
{
    this->id_ = id;
    this->current_time_ = start_value;
}

GPU CPU int timer_t::get_id() const
{
    return this->id_;
}

GPU CPU double timer_t::get_time() const
{
    return this->current_time_();
}

GPU void timer_t::set_time(const double new_value)
{
    this->current_time_ = new_value;
}

GPU void timer_t::add_time(const double progression)
{
    this->current_time_ += progression;
}

GPU timer_t timer_t::duplicate() const
{
    return timer_t{this->id_, this->current_time_};
}


