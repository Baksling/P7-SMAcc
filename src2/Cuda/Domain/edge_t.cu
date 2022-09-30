#include "edge_t.h"


edge_t::edge_t(const int id, const float weight, node_t* dest, constraint_t* guard)
{
    this->id_ = id;
    this->dest_ = dest;
    this->weight_ = weight;
    this->updates_ = array_t<update_t>(0);
    this->guard_ = guard;
}

GPU float edge_t::get_weight() const
{
    return this->weight_;
}

void edge_t::set_updates(std::list<update_t>* updates)
{
    this->updates_ = to_array(updates);
}

GPU bool edge_t::evaluate_constraints(const lend_array<clock_timer_t>* timers) const
{
    if(this->guard_ == nullptr) return true;
    return this->guard_->evaluate(timers);
}
