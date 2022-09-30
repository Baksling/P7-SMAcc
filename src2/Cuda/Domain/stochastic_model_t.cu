#include "stochastic_model_t.h"

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
