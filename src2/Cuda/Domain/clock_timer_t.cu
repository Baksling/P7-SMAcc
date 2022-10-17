#include "clock_timer_t.h"

CPU GPU clock_timer_t::clock_timer_t(const int id, const double start_value)
{
    this->id_ = id;
    this->current_time_ = start_value;
    this->temp_time_ = start_value;
}

int clock_timer_t::get_id() const
{
    return this->id_;
}

CPU GPU double clock_timer_t::get_time() const
{
    return this->current_time_;
}

CPU GPU double clock_timer_t::get_temp_time() const
{
    return this->temp_time_;
}

void clock_timer_t::set_temp_time(const double new_value)
{
    this->temp_time_ = new_value;
}

void clock_timer_t::reset_temp_time()
{
    this->temp_time_ = this->current_time_;
}

CPU GPU void clock_timer_t::set_time(const double new_value)
{
    this->current_time_ = new_value;
    set_temp_time(new_value);
}

CPU GPU void clock_timer_t::add_time(const double progression)
{
    this->current_time_ += progression;
    set_temp_time(this->current_time_);
}

CPU GPU clock_timer_t clock_timer_t::duplicate() const
{
    return clock_timer_t{this->id_, this->current_time_};
}

// ReSharper disable once CppMemberFunctionMayBeStatic
void clock_timer_t::accept(visitor* v)
{
    return;
    //v.visit()
}

void clock_timer_t::cuda_allocate(clock_timer_t** pointer, const allocation_helper* helper) const
{
    cudaMalloc(pointer, sizeof(clock_timer_t));
    helper->free_list->push_back(*pointer);
    cudaMemcpy(*pointer, this, sizeof(clock_timer_t), cudaMemcpyHostToDevice);
}


