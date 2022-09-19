//
// Created by Patrick on 19-09-2022.
//

#define GPU __device__
#define CPU __host__

#include "guard_d.h"

CPU guard_d::guard_d(int id, int timer_id, logical_operator type, double value) {
    this->id_ = id;
    this->timer_id_ = timer_id;
    this->type_ = type;
    this->value_ = value;
}

GPU int guard_d::get_id() {
    return this->id_;
}

GPU int guard_d::get_timer_id() {
    return this->timer_id_;
}

GPU logical_operator guard_d::get_type() {
    return this->type_;
}

GPU bool guard_d::validate(double value) {
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