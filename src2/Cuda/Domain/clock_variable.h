#pragma once

#ifndef CLOCK_TIMER_T_H
#define CLOCK_TIMER_T_H

// #include "common.h"
#include "../common/macro.h"
#include "../common/allocation_helper.h"
#include "../Visitors/visitor.h"

class clock_variable {
private:
    int id_;
    double current_time_;
    double temp_time_;
    double max_value_;
public:
    CPU GPU explicit clock_variable(int id, double start_value);

    //SIMULATOR METHODS
    CPU GPU double get_time() const;
    CPU GPU double get_temp_time() const;
    CPU GPU void set_temp_time(double new_value);
    CPU GPU void reset_temp_time();
    CPU GPU void set_time(double new_value);
    CPU GPU void add_time(double progression);
    CPU GPU clock_variable duplicate() const;
    CPU GPU double get_max_value() const;
    
    //HOST METHODS
    void accept(visitor* v);
    void pretty_print() const;
    void cuda_allocate(clock_variable* pointer, const allocation_helper* helper) const;
};

#endif
