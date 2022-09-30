#include "node_t.h"

node_t::node_t(const int id, constraint_t* invariant, const bool is_goal, const bool is_branch)
{
    this->id_ = id;
    this->is_goal_ = is_goal;
    this->is_branch_ = is_branch;
    this->invariant_ = invariant;
    this->edges_ = array_t<edge_t>(0);
}

void node_t::set_edges(std::list<edge_t>* list)
{
    this->edges_ = to_array(list);
}

GPU lend_array<edge_t> node_t::get_edges()
{
    return lend_array<edge_t>(&this->edges_);
}

GPU bool node_t::is_goal_node() const
{
    return this->is_goal_;
}


GPU bool node_t::evaluate_invariants(const lend_array<timer_t>* timers) const
{
    if(this->invariant_ == nullptr) return true;
    return this->invariant_->evaluate(timers);
}

void node_t::accept(visistor& v)
{
    const lend_array<edge_t> edges = this->get_edges();
    for (int i = 0; i < edges.size(); ++i)
    {
        v.visit(edges.at(i));
    }
    v.visit(this->invariant_);
}

int node_t::get_id() const
{
    return this->id_;
}

bool node_t::is_branch() const
{
    return this->is_branch_;
}
