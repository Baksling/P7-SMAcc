#include "Node.h"

#include <functional>

using namespace std;

node::node(const int id) {
    this->id = id;
}

int node::get_id() const
{
    return id;
}
