//
// Created by Patrick on 19-09-2022.
//
#define GPU __device__
#define CPU __host__

#ifndef SRC2_SLN_EDGE_D_H
#define SRC2_SLN_EDGE_D_H


class edge_d {
private:
    int id_;
    int dest_node_;
public:
    CPU edge_d(int id, int dest_node_id);
    GPU int get_id();
    GPU int get_dest_node();
};


#endif //SRC2_SLN_EDGE_D_H
