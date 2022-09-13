#include "Timer.h"

timer::timer(const double time)
{
    this->time_ = time;
}

double timer::get_time()
{
    return this->time_;
}

