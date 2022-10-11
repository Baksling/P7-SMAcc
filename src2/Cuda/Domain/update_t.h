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

    //SIMULATOR METHODS
    CPU GPU void apply_update(const lend_array<clock_timer_t>* timers) const;
    
    //HOST METHODS
    int get_id() const;
    CPU GPU int get_timer_id() const;
    CPU GPU float get_timer_value() const;
    void accept(visitor* v);
    void cuda_allocate(update_t** pointer, const allocation_helper* helper) const;
};

#endif