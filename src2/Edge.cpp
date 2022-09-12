#include "Edge.h"
#include "Node.h"
using namespace std;

edge::edge(node* n1, node* n2, const float weight) {
    n1_ = n1;
    n2_ = n2;
    weight_ = weight;
}

node* edge::get_n1() const
{
    return n1_;
}

node* edge::get_n2() const
{
    return n2_;
}

float edge::get_weight() const
{
    return weight_;
}
