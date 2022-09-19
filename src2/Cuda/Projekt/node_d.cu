//
// Created by Patrick on 19-09-2022.
//

#define GPU __device__
#define CPU __host__

#include "node_d.h"

node_d::node_d(const int id, const bool is_goal) {
    this->id_ = id;
    this->is_goal_ = is_goal;
}

GPU int node_d::get_id() {
    return this->id_;
}

GPU bool node_d::is_goal() {
    return this->is_goal_;
}