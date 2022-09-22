//
// Created by Patrick on 19-09-2022.
//

#define GPU __device__
#define CPU __host__

#include "timer_d.h"
#include <cuda.h>
#include <cuda_runtime.h>


CPU GPU timer_d::timer_d(int id, double start_value) {
    this->id_ = id;
    this->value_ = start_value;
}

GPU double timer_d::get_value() {
    return this->value_;
}

GPU void timer_d::set_value(double new_value) {
    this->value_ = new_value;
}

GPU timer_d timer_d::copy()
{
    return timer_d(this->id_, this->value_);
}
