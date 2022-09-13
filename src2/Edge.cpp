#include "Edge.h"
#include "Node.h"
using namespace std;

edge::edge(node* n1, const list<guard>& guards) {
    node_ = n1;
    this->guards_ = guards;
}

node* edge::get_node() const
{
    return node_;
}

float edge::get_weight() const
{
    return weight_;
}

void edge::add_guard(guard guard)
{
    this->guards_.push_back(guard);
}

bool edge::validate()
{
    for (guard guard : this->guards_)
    {
        if (!guard.validate_guard()) return false;
    }
    
    if (!this->get_node()->validate_invariants()) return false; 

    return true;
}

