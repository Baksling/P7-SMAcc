//
// Created by Patrick on 19-09-2022.
//

#define GPU __device__
#define CPU __host__

#include "guard_d.h"

#include <stdio.h>

guard_d::guard_d(const int timer_id, const logical_operator type, const double value, const int id) {
    this->timer_id_ = timer_id;
    this->type_ = type;
    this->value_ = value;
    this->id_ = id;
}

CPU GPU int guard_d::get_timer_id() const
{
    return this->timer_id_;
}

CPU GPU logical_operator guard_d::get_type() const
{
    return this->type_;
}

CPU GPU double guard_d::get_value() const
{
    return this->value_;
}

GPU bool guard_d::validate(const double value) const
{
    //printf("Validating %d, value: %f, target %f %d id: %d \n", this->timer_id_, value, this->value_, this->type_, this->id_);
    switch (this->type_) {
        case logical_operator::greater_equal: return value >= this->value_;
        case logical_operator::less_equal: return value <= this->value_;
        case logical_operator::greater: return value > this->value_;
        case logical_operator::less: return value < this->value_;
        case logical_operator::equal: return value == this->value_;
        case logical_operator::not_equal: return value != this->value_;
    }
    return false;
}
