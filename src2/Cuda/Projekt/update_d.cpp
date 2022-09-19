//
// Created by Patrick on 19-09-2022.
//

#define GPU __device__
#define CPU __host__

#include "update_d.h"

CPU update_d::update_d(int id, int timer_id, double value) {
    this->id_ = id;
    this->timer_id_ = timer_id;
    this->value_ = value;
}

GPU int update_d::get_id() {
    return this->id_;
}

GPU int update_d::get_timer_id() {
    return this->timer_id_;
}

GPU double update_d::get_value() {
    return this->value_;
}