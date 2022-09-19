//
// Created by Patrick on 19-09-2022.
//

#define GPU __device__
#define CPU __host__

#ifndef SRC2_SLN_TIMER_D_H
#define SRC2_SLN_TIMER_D_H


class timer_d {
private:
    int id_;
    double value_;
public:
    CPU timer_d(int id, int start_value);
    GPU double get_value();
    GPU void set_value(double new_value);
};


#endif //SRC2_SLN_TIMER_D_H
