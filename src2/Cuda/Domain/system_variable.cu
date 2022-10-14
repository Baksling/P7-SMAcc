#include "system_variable.h"

CPU GPU system_variable::system_variable(int id, int initial_value)
{
    this->id_ = id;
    this->value_ = initial_value;
}

CPU GPU int system_variable::get_value() const
{
    return this->value_;
}

CPU GPU void system_variable::set_value(const int new_value)
{
    this->value_ = new_value;
}

CPU GPU system_variable system_variable::duplicate() const
{
    return system_variable{this->id_, this->value_};
}

int system_variable::get_id() const
{
    return this->id_;
}

void system_variable::accept(visitor* v)
{
    //TODO fix this
    return;
}

void system_variable::cuda_allocate(system_variable** p, const allocation_helper* helper)
{
    return; //fix this
}
