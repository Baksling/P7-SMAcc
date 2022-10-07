#include "node_t.h"

#include "CudaSimulator.h"

node_t::node_t(node_t* source, constraint_t* invariant, array_t<edge_t*> edges)
{
    this->id_ = source->id_;
    this->is_branch_point_ = source->is_branch_point_;
    this->invariant_ = invariant;
    this->is_goal_ = source->is_goal_;
    this->edges_ = edges;
}

node_t::node_t(const int id, const bool is_branch_point, constraint_t* invariant, const bool is_goal)
{
    this->id_ = id;
    this->is_goal_ = is_goal;
    this->invariant_ = invariant;
    this->is_branch_point_ = is_branch_point;
    this->edges_ = array_t<edge_t*>(0);
}

GPU CPU int node_t::get_id() const
{
    return this->id_;
}

void node_t::set_edges(std::list<edge_t*>* list)
{
    this->edges_ = to_array(list);
}

CPU GPU lend_array<edge_t*> node_t::get_edges()
{
    return lend_array<edge_t*>(&this->edges_);
}

CPU GPU bool node_t::is_goal_node() const
{
    return this->is_goal_;
}

GPU bool node_t::evaluate_invariants(const lend_array<clock_timer_t>* timers) const
{
    if(this->invariant_ == nullptr) return true;
    return this->invariant_->evaluate(timers);
}

void node_t::accept(visitor* v)
{
    const lend_array<edge_t*> edges = this->get_edges();
    v->visit(this->invariant_);
    for (int i = 0; i < edges.size(); ++i)
    {
        v->visit(*edges.at(i));
    }
    for (int i = 0; i < edges.size(); ++i)
    {
        v->visit(edges.get(i)->get_dest());
    }
}

void node_t::cuda_allocate(node_t** pointer, const allocation_helper* helper)
{
    if(helper->node_map->count(this) == 1) return;
    cudaMalloc(pointer, sizeof(node_t));
    helper->free_list->push_back(*pointer);
    helper->node_map->insert(std::pair<node_t*, node_t*>(this, *pointer) );
    
    std::list<edge_t*> edge_lst;
    for (int i = 0; i < this->edges_.size(); ++i)
    {
        edge_t* edge_p = nullptr;
        this->edges_.get(i)->cuda_allocate(&edge_p, helper);
        edge_lst.push_back(edge_p);
    }
    constraint_t* invariant_p = nullptr;
    if (this->invariant_ != nullptr)
    {
        this->invariant_->cuda_allocate(&invariant_p, helper);
    }
    
    const node_t result(this, invariant_p, cuda_to_array(&edge_lst, helper->free_list));
    cudaMemcpy(*pointer, &result, sizeof(node_t), cudaMemcpyHostToDevice);
}

CPU GPU bool node_t::is_branch_point() const
{
    return this->is_branch_point_;
}

GPU double node_t::max_time_progression(const lend_array<clock_timer_t>* timers, double max_progression) const
{
    if(this->invariant_ == nullptr)
    {
        return max_progression;
    }
    
    return this->invariant_->max_time_progression(timers, max_progression); 
}
