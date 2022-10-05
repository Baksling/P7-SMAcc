#pragma once

#ifndef UPDATE_T_H
#define UPDATE_T_H

#include "common.h"

class update_t
{
private:
    int id_;
    int timer_id_;
    double timer_value_;
public:
    update_t(int id, int timer_id, double timer_value);
    GPU void update_timer(const lend_array<clock_timer_t>* timers) const;
    void accept(visitor* v);
    int get_timer_id() const;
    float get_timer_value() const;
    int get_id() const;
    void cuda_allocate(update_t** pointer, std::list<void*>* free_list);
};

#endif