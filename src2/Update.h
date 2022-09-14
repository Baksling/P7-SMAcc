#ifndef UPDATE_H
#define UPDATE_H

#include "Timer.h"

class update
{
private:
    timer* timer_;
    double value_;
public:
    update(timer* timer, double value);
    void activate();
};

#endif // UPDATE_H
