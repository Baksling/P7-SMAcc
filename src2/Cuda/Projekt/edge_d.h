//
// Created by Patrick on 19-09-2022.
//
#define GPU __device__
#define CPU __host__

#ifndef SRC2_SLN_EDGE_D_H
#define SRC2_SLN_EDGE_D_H

#include <cuda.h>
#include <cuda_runtime.h>

class edge_d {
private:
    int id_;
    int dest_node_;
    int weight_;
public:
    edge_d(int id, int dest_node_id, int weight = 1);
    CPU GPU int get_id();
    CPU GPU int get_dest_node();
};


#endif //SRC2_SLN_EDGE_D_H
