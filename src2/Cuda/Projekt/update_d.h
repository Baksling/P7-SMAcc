//
// Created by Patrick on 19-09-2022.
//

#define GPU __device__
#define CPU __host__

#ifndef SRC2_SLN_UPDATE_D_H
#define SRC2_SLN_UPDATE_D_H


class update_d {
private:
    int id_;
    int timer_id_;
    double value_;
public:
    CPU update_d(int id, int timer_id, double value);
    GPU int get_id();
    GPU int get_timer_id();
    GPU double get_value();
};


#endif //SRC2_SLN_UPDATE_D_H
