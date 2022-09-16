#include "Node.h"
#include <iostream>
#include <functional>
#include "Update.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

node::node(const int id, const bool is_goal) {
    this->id_ = id;
    this->is_goal_ = is_goal;
    
    const list<edge> temp;
    edges_ = temp;
}

__host__ void node::add_edge(node* n, const list<guard> guards, list<update>* updates)
{
    this->edges_.emplace_back(n, guards, updates);
}

__device__ int node::get_id()
{
    return id_;
}

__host__ bool node::is_goal() const
{
    return is_goal_;
}

__host__ void node::add_invariant(const logical_operator type, const double value, timer* timer)
{
    this->invariants_.emplace_back(type, value, timer);
}

__host__ list<edge>* node::get_edges()
{
    return &this->edges_;
}

__host__ bool node::validate_invariants()
{
    for (guard guard : this->invariants_)
    {
        if (!guard.validate_guard()) return false;
    }

    return true;
}

__host__ list<guard>* node::get_invariants()
{
    return &this->invariants_;
}


