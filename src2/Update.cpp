﻿#include "Update.h"

update::update(timer* timer, const double value)
{
    this->timer_ = timer;
    this->value_ = value;
}

void update::activate()
{
    this->timer_->set_time(this->value_);
}

