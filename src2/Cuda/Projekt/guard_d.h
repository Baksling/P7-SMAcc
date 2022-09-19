//
// Created by Patrick on 19-09-2022.
//

#define GPU __device__
#define CPU __host__

#ifndef SRC2_SLN_GUARD_D_H
#define SRC2_SLN_GUARD_D_H

enum logical_operator
{
    less_equal = 0,
    greater_equal,
    less,
    greater,
    equal,
    not_equal
};


class guard_d {
private:
    int id_;
    int timer_id_;
    logical_operator type_;
    double value_;
public:
    CPU guard_d(int id, int timer_id, logical_operator type, double value);
    GPU int get_id();
    GPU int get_timer_id();
    GPU logical_operator get_type();
    GPU bool validate(double value);
};


#endif //SRC2_SLN_GUARD_D_H
