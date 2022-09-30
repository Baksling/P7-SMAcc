#pragma once
#include "common.h"
#include "timer_t.h"

class update_t
{
private:
    int id_;
    int timer_id_;
    double timer_value_;
public:
    update_t(int id, int timer_id, double timer_value);
    GPU void update_timer(const lend_array<timer_t>* timers) const;
};
