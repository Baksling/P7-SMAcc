//
// Created by Patrick on 19-09-2022.
//

#ifndef SRC2_SLN_UPDATE_D_H
#define SRC2_SLN_UPDATE_D_H
#include "common.h"


class update_d {
private:
    int timer_id_;
    double value_;
public:
    update_d(int timer_id, double value);
    CPU GPU int get_timer_id();
    CPU GPU double get_value();
};


#endif //SRC2_SLN_UPDATE_D_H
