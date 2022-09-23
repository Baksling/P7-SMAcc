//
// Created by Patrick on 19-09-2022.
//

#define GPU __device__
#define CPU __host__

#ifndef SRC2_SLN_TIMER_D_H
#define SRC2_SLN_TIMER_D_H
#include <cuda.h>
#include <cuda_runtime.h>


class timer_d {
private:
    int id_;
    double value_;
public:
    CPU GPU timer_d(int id, double start_value);
    GPU double get_value() const;
    GPU void set_value(double new_value);
    GPU timer_d copy() const;
};


#endif //SRC2_SLN_TIMER_D_H
