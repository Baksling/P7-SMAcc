#include "system_variable.h"

CPU GPU system_variable::system_variable(const int id, const int initial_value)
{
    this->id_ = id;
    this->value_ = initial_value;
    this->temp_value_ = initial_value;
}

CPU GPU int system_variable::get_value() const
{
    return this->value_;
}

CPU GPU void system_variable::reset_temp()
{
    this->temp_value_ = this->value_;
}

int system_variable::get_temp_value() const
{
    return this->temp_value_;
}

void system_variable::set_temp_value(const int temp_value)
{
    this->temp_value_ = temp_value;
}

void system_variable::set_value(const int new_value)
{
    this->value_ = new_value;
    this->temp_value_ = new_value;
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

void system_variable::cuda_allocate(system_variable* p, const allocation_helper* helper) const
{
    cudaMemcpy(p, this, sizeof(system_variable), cudaMemcpyHostToDevice);
}
