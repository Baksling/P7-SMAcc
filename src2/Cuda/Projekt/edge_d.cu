//
// Created by Patrick on 19-09-2022.
//
#define GPU __device__
#define CPU __host__

#include "edge_d.h"

edge_d::edge_d(int id, int dest_node_id) {
    this->id_ = id;
    this->dest_node_ = dest_node_id;
}

CPU GPU int edge_d::get_id() {
    return this->id_;
}

CPU GPU int edge_d::get_dest_node() {
    return this->dest_node_;
}