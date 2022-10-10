#pragma once

#ifndef CLOCK_TIMER_T_H
#define CLOCK_TIMER_T_H

#include "common.h"

class visitor;

class clock_timer_t {
private:
    int id_;
    double current_time_;
public:
    CPU GPU explicit clock_timer_t(int id, double start_value);

    //SIMULATOR METHODS
    CPU GPU double get_time() const;
    CPU GPU void set_time(double new_value);
    CPU GPU void add_time(double progression);
    CPU GPU clock_timer_t duplicate() const;
    
    //HOST METHODS
    int get_id() const;
    void accept(visitor* v);
    void cuda_allocate(clock_timer_t** pointer, const allocation_helper* helper) const;
};

#endif