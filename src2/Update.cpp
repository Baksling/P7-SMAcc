#include "Update.h"

update::update(timer* timer, const double value)
{
    this->timer_ = timer;
    this->value_ = value;
}

update::update(int timer_id, double value)
{
    this->value_ = value;
}


void update::activate()
{
    this->timer_->set_time(this->value_);
}

