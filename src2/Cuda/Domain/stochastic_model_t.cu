#include "stochastic_model_t.h"
#include <cassert>

#include "../../Timer.h"

stochastic_model_t::stochastic_model_t(node_t* start_node, array_t<timer_t>* timers)
{
    this->start_node_ = start_node;
    this->timers_ = timers;
}
void stochastic_model_t::accept(visistor& v)
{
    v.visit(this->start_node_);
    for (int i = 0; i < this->timers_->size(); ++i)
    {
        v.visit(&this->timers_[i]);
    }
}

array_t<timer_t> stochastic_model_t::create_internal_timers() const
{
    const int size = this->timers_->size();
    timer_t* internal_timers_arr = static_cast<timer_t*>(malloc(sizeof(timer_t) * size));
    
    
    for (int i = 0; i < size; i++)
    {
        internal_timers_arr[i] = this->timers_->at(i)->duplicate();
    }

    const array_t<timer_t> internal_timers{ internal_timers_arr, size};
    return internal_timers;
}

void stochastic_model_t::reset_timers(array_t<timer_t>* active_timers) const
{
    assert(active_timers->size() == this->timers_->size());
    for (int i = 0; i < active_timers->size(); i++)
    {
        timer_t* timer = active_timers->at(i);
        timer->set_time(this->timers_->at(i)->get_time());
    }
}

node_t* stochastic_model_t::get_start_node() const
{
    return this->start_node_;
}
