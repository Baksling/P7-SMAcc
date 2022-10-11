#include "node_t.h"

node_t::node_t(node_t* source, const array_t<constraint_t*> invariant, const array_t<edge_t*> edges)
{
    this->id_ = source->id_;
    this->is_branch_point_ = source->is_branch_point_;
    this->invariants_ = invariant;
    this->is_goal_ = source->is_goal_;
    this->edges_ = edges;
}

node_t::node_t(const int id, const array_t<constraint_t*> invariants, const bool is_branch_point, const bool is_goal)
{
    this->id_ = id;
    this->is_goal_ = is_goal;
    this->invariants_ = invariants;
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

CPU GPU bool node_t::evaluate_invariants(const lend_array<clock_timer_t>* timers) const
{
    for (int i = 0; i < this->invariants_.size(); ++i)
    {
        if(!this->invariants_.get(i)->evaluate(timers))
            return false;
    }

    return true;
}

void node_t::accept(visitor* v) const
{
    //visit node constraints
    for (int i = 0; i < this->invariants_.size(); ++i)
    {
        printf("    ");
        v->visit(this->invariants_.get(i));
    }

    //visit edges
    for (int i = 0; i < this->edges_.size(); ++i)
    {
        v->visit(this->edges_.get(i));
    }

    //visit edge destinations
    for (int i = 0; i < this->edges_.size(); ++i)
    {
        v->visit(this->edges_.get(i)->get_dest());
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

    std::list<constraint_t*> invariant_lst;
    for (int i = 0; i < this->invariants_.size(); ++i)
    {
        constraint_t* invariant_p = nullptr;
        this->invariants_.get(i)->cuda_allocate(&invariant_p, helper);
        invariant_lst.push_back(invariant_p);
    }
    
    const node_t result(this,
        cuda_to_array(&invariant_lst, helper->free_list),
        cuda_to_array(&edge_lst, helper->free_list));
    cudaMemcpy(*pointer, &result, sizeof(node_t), cudaMemcpyHostToDevice);
}


void node_t::cuda_allocate_2(node_t* cuda_p, const allocation_helper* helper) const
{
    edge_t* edges = nullptr; 
    cudaMalloc(&edges,sizeof(edge_t)*this->edges_.size());
    helper->free_list->push_back(edges);

    for (int i = 0; i < this->edges_.size(); ++i)
    {
        this->edges_.get(i)->cuda_allocate_2(&edges[i], helper);
    }

    constraint_t* invariants = nullptr; 
    cudaMalloc(&invariants,sizeof(constraint_t)*this->invariants_.size());
    helper->free_list->push_back(cuda_p);
    for (int i = 0; i < this->invariants_.size(); ++i)
    {
        this->invariants_.get(i)->cuda_allocate_2(&invariants[i], helper);
    }
    
    // const node_t result(this,
    //     array_t<constraint_t>(invariants, this->invariants_.size()),
    //     array_t<edge_t>(edges, this->edges_.size()));
    
    // cudaMemcpy(cuda_p, &result, sizeof(node_t), cudaMemcpyHostToDevice);
}

CPU GPU bool node_t::is_branch_point() const
{
    return this->is_branch_point_;
}

CPU GPU double node_t::max_time_progression(const lend_array<clock_timer_t>* timers, double max_progression) const
{
    if(this->invariants_.size() <= 0) return max_progression;

    for (int i = 0; i < this->invariants_.size(); ++i)
    {
        const double temp_progression = this->invariants_.get(i)->max_time_progression(timers, max_progression);
        max_progression = temp_progression < max_progression ? temp_progression : max_progression;
    }
    
    return max_progression; 
}
