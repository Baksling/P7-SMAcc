#pragma once

#ifndef UPDATE_T_H
#define UPDATE_T_H

#include "common.h"
#include "clock_timer_t.h"

class update_t : public element
{
private:
    int id_;
    int timer_id_;
    double timer_value_;
public:
    update_t(int id, int timer_id, double timer_value);
    GPU void update_timer(const lend_array<clock_timer_t>* timers) const;
    void accept(visistor& v) override;
    int get_timer_id() const;
    float get_timer_value() const;
    int get_id() const;
};

#endif