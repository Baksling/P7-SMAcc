#include "Edge.h"
#include "Node.h"
#include <iostream>
using namespace std;

edge::edge(node* n1, const list<guard>& guards, list<update>* updates) {
    this->node_ = n1;
    this->guards_ = guards;
    this->updates_ = updates;
}

edge::edge()
{
    
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

void edge::activate()
{
    if (this->updates_->empty()) return;
    for (update update : *this->updates_)
    {
        update.activate();
    }
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

