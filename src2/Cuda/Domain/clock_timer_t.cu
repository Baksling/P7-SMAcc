#include "clock_timer_t.h"

CPU GPU clock_timer_t::clock_timer_t(const int id, const double start_value)
{
    this->id_ = id;
    this->current_time_ = start_value;
}

GPU CPU int clock_timer_t::get_id() const
{
    return this->id_;
}

GPU CPU double clock_timer_t::get_time() const
{
    return this->current_time_;
}

GPU void clock_timer_t::set_time(const double new_value)
{
    this->current_time_ = new_value;
}

GPU void clock_timer_t::add_time(const double progression)
{
    this->current_time_ += progression;
}

GPU clock_timer_t clock_timer_t::duplicate() const
{
    return clock_timer_t{this->id_, this->current_time_};
}

void clock_timer_t::accept(visitor* v)
{
    //v.visit()
}


