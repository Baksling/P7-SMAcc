//
// Created by Patrick on 19-09-2022.
//

#define GPU __device__
#define CPU __host__

#include "timer_d.h"
#include <cuda.h>
#include <cuda_runtime.h>


CPU GPU timer_d::timer_d(const int id, const double start_value) {
    this->id_ = id;
    this->value_ = start_value;
}

int timer_d::get_id() const
{
    return this->id_;
}

GPU double timer_d::get_value() const
{
    return this->value_;
}

GPU void timer_d::set_value(const double new_value) {
    this->value_ = new_value;
}

GPU void timer_d::add_time(const double progression) {
    this->value_ += progression;
}

GPU timer_d timer_d::copy() const
{
    return timer_d(this->id_, this->value_);
}
