#include "update_t.h"

update_t::update_t(const int id, const int timer_id, const double timer_value)
{
    this->id_ = id;
    this->timer_id_ = timer_id;
    this->timer_value_ = timer_value;
}

GPU void update_t::update_timer(const lend_array<timer_t>* timers) const
{
    timers->at(this->timer_id_)->set_time(this->timer_value_);
}
