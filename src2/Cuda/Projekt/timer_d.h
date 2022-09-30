//
// Created by Patrick on 19-09-2022.
//

#ifndef SRC2_SLN_TIMER_D_H
#define SRC2_SLN_TIMER_D_H
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"


class timer_d {
private:
    int id_;
    double value_;
public:
    CPU GPU timer_d(int id, double start_value);
    CPU GPU int get_id() const;
    CPU GPU double get_value() const;
    GPU void set_time(double new_value);
    GPU void add_time(double progression);
    GPU timer_d copy() const;
};


#endif //SRC2_SLN_TIMER_D_H
