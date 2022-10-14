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
    CPU GPU explicit system_variable(int id, int initial_value = 0);

    //SIMULATION methods
    CPU GPU int get_value() const;
    CPU GPU void set_value(int new_value);
    CPU GPU system_variable duplicate() const;
    
    //HOST methods
    void accept(visitor* v);
    CPU GPU void cuda_allocate(system_variable** p, const allocation_helper* helper);
};

#endif