#include "clock_variable.h"

clock_variable::clock_variable(const int id, const double start_value)
{
    this->id_ = id;
    this->current_time_ = start_value;
    this->temp_time_ = start_value;
    this->max_value_ = start_value;
}


int clock_variable::get_id() const
{
    return this->id_;
}

CPU GPU double clock_variable::get_time() const
{
    return this->current_time_;
}

CPU GPU double clock_variable::get_temp_time() const
{
    return this->temp_time_;
}

void clock_variable::set_temp_time(const double new_value)
{
    this->temp_time_ = new_value;
}

void clock_variable::reset_temp_time()
{
    this->temp_time_ = this->current_time_;
}

CPU GPU void clock_variable::set_time(const double new_value)
{
    this->current_time_ = new_value;
    this->temp_time_ = new_value;
    this->max_value_ = this->max_value_ > new_value ? this->max_value_ : new_value;
}

CPU GPU void clock_variable::add_time(const double progression)
{
    set_time(this->current_time_ + progression);
}

CPU GPU clock_variable clock_variable::duplicate() const
{
    return clock_variable{this->id_, this->current_time_};
}

// ReSharper disable once CppMemberFunctionMayBeStatic
void clock_variable::accept(visitor* v)
{
    return;
    //v.visit()
}

void clock_variable::cuda_allocate(clock_variable* pointer, const allocation_helper* helper) const
{
    cudaMemcpy(pointer, this, sizeof(clock_variable), cudaMemcpyHostToDevice);
}


