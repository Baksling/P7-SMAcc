#include "Node.h"

#include <functional>

using namespace std;

node::node(const int id, const bool is_goal) {
    this->id = id;
    this->is_goal_ = is_goal;
    
    const list<edge*> temp;
    edges_ = temp;
}

void node::add_edge(edge* e)
{
    edges_.push_back(e);
}

void node::add_edge(node* n, const float weight)
{
    edge* temp = new edge(n, weight);
    this->add_edge(temp);
}

int node::get_id() const
{
    return id;
}

bool node::is_goal() const
{
    return is_goal_;
}

list<edge*> node::get_edges()
{
    return edges_;
}
