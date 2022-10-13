#pragma once

#ifndef SYSTEM_VARIABLE_H
#define SYSTEM_VARIABLE_H

#include "common.h"

class system_variable
{
private:
    int value_;
    int id_;
public:
    explicit system_variable(int id, int initial_value = 0);
    CPU GPU int get_value() const;
    CPU GPU int set_value(int new_value);
    void accept(visitor* v);
    
    CPU GPU void cuda_allocate(system_variable** p, allocation_helper* helper);
};

#endif