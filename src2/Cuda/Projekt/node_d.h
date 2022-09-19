//
// Created by Patrick on 19-09-2022.
//

#define GPU __device__
#define CPU __host__

#ifndef SRC2_SLN_NODE_D_H
#define SRC2_SLN_NODE_D_H


class node_d {
private:
    int id_;
    bool is_goal_;
public:
    CPU node_d(int id, bool is_goal);
    GPU int get_id();
    GPU bool is_goal();
};


#endif //SRC2_SLN_NODE_D_H
