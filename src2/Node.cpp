#include "Node.h"

#include <functional>

using namespace std;

node::node(const int id, const bool is_goal) {
    this->id_ = id;
    this->is_goal_ = is_goal;
    
    const list<edge> temp;
    edges_ = temp;
}

void node::add_edge(node* n, const float weight)
{
    this->edges_.emplace_back(n, weight);
}

int node::get_id() const
{
    return id_;
}

bool node::is_goal() const
{
    return is_goal_;
}

void node::add_guard(const logical_operator type, const double value, timer* timer)
{
    this->invariants_.emplace_back(type, value, timer);
}

list<edge> node::get_edges()
{
    return edges_;
}

bool node::validate_invariants()
{
    for (guard guard : this->invariants_)
    {
        if (!guard.validate_guard()) return false;
    }

    return true;
}


