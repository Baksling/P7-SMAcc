#include "Edge.h"
#include "Node.h"
using namespace std;

edge::edge(node* n1, const float weight) {
    node_ = n1;
    weight_ = weight;
}

node* edge::get_node() const
{
    return node_;
}

float edge::get_weight() const
{
    return weight_;
}

bool edge::validate()
{
    //Placeholder! ;)
    return true;
}

