﻿#include "update_t.h"

update_t::update_t(const int id, const int timer_id, const double timer_value)
{
    this->id_ = id;
    this->timer_id_ = timer_id;
    this->timer_value_ = timer_value;
}

GPU void update_t::update_timer(const lend_array<clock_timer_t>* timers) const
{
    timers->at(this->timer_id_)->set_time(this->timer_value_);
}

void update_t::accept(visitor* v)
{
    return;
}

int update_t::get_timer_id() const
{
    return this->timer_id_;
}

float update_t::get_timer_value() const
{
    return static_cast<float>(this->timer_value_);
}

int update_t::get_id() const
{
    return this->id_;
}

void update_t::cuda_allocate(update_t** pointer, std::list<void*>* free_list)
{
    cudaMalloc(pointer, sizeof(update_t));
    free_list->push_back(*pointer);
    cudaMemcpy(*pointer, this, sizeof(update_t), cudaMemcpyHostToDevice);
}
