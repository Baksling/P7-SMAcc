#include "update_t.h"

double update_t::evaluate_expression()
{
    return 0.0;
}

update_t::update_t(const int id, const int timer_id, const bool is_clock_update, update_expression* expression)
{
    this->id_ = id;
    this->timer_id_ = timer_id;
    this->clock_update_ = is_clock_update;
    this->expression_ = expression;

    const unsigned int expression_depth = expression->get_depth() + 1;
    this->evaluation_stack_ = cuda_stack<update_expression*>(expression_depth);
    this->value_stack_ = cuda_stack<double>(expression_depth);
}

CPU GPU void update_t::apply_update(const lend_array<clock_timer_t>* timers, const lend_array<system_variable>* variables) const
{
    const timer*
    
    timers->at(this->timer_id_)->set_time(this->timer_value_);
}

void update_t::accept(visitor* v)
{
    return;
}

CPU GPU int update_t::get_timer_id() const
{
    return this->timer_id_;
}

CPU GPU float update_t::get_timer_value() const
{
    return static_cast<float>(this->timer_value_);
}

int update_t::get_id() const
{
    return this->id_;
}

void update_t::cuda_allocate(update_t** pointer, const allocation_helper* helper) const
{
    cudaMalloc(pointer, sizeof(update_t));
    helper->free_list->push_back(*pointer);
    cudaMemcpy(*pointer, this, sizeof(update_t), cudaMemcpyHostToDevice);
}
