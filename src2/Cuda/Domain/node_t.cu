#include "node_t.h"

#include <string>

#include "edge_t.h"
#include "simulator_state.h"

node_t::node_t(const node_t* source, const array_t<constraint_t*> invariant, const array_t<edge_t*> edges, expression* lambda)
{
    this->id_ = source->id_;
    this->is_branch_point_ = source->is_branch_point_;
    this->invariants_ = invariant;
    this->is_goal_ = source->is_goal_;
    this->edges_ = edges;
    this->lambda_expression_ = lambda;
}


node_t::node_t(const int id, const array_t<constraint_t*> invariants,
    const bool is_branch_point, const bool is_goal, expression* lambda)
{
    if(lambda == nullptr)
    {
        lambda = expression::literal_expression(1);
    }
    
    this->id_ = id;
    this->is_goal_ = is_goal;
    this->invariants_ = invariants;
    this->lambda_expression_ = lambda;
    this->is_branch_point_ = is_branch_point;
    this->edges_ = array_t<edge_t*>(0);
}

GPU CPU int node_t::get_id() const
{
    return this->id_;
}

GPU CPU double node_t::get_lambda(simulator_state* state) const
{
    if(this->lambda_expression_ != nullptr)
        return state->evaluate_expression(this->lambda_expression_);

    return 0.0; //illegal lambda value.
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

CPU GPU bool node_t::evaluate_invariants(simulator_state* state) const
{
    for (int i = 0; i < this->invariants_.size(); ++i)
    {
        if(!this->invariants_.get(i)->evaluate(state))
            return false;
    }

    return true;
}

void node_t::accept(visitor* v) const
{
    //visit node constraints
    if(this->lambda_expression_ != nullptr)
        v->visit(this->lambda_expression_);
    
    for (int i = 0; i < this->invariants_.size(); ++i)
    {
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

void node_t::pretty_print(std::ostream& os) const
{
    os << "\nNode id: " + std::to_string(this->id_) + " | Is branch: " + std::to_string(this->is_branch_point()) + " | Is goal: "
        + std::to_string(this->is_goal_) + "\n";
    // printf("\nNode id: %3d | Is branch: %d | Is goal: %d \n", this->id_, this->is_branch_point_,
    //        this->is_goal_);
}

void node_t::cuda_allocate(node_t* pointer, const allocation_helper* helper)
{
    if(helper->node_map->count(this) == 1) return;
    // cudaMalloc(pointer, sizeof(node_t));
    // helper->free_list->push_back(*pointer);

    //add current node to circular reference resolver
    helper->node_map->insert(std::pair<node_t*, node_t*>(this, pointer) );
    
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

    expression* expr = nullptr;
    if(this->lambda_expression_ != nullptr)
    {
        cudaMalloc(&expr, sizeof(expression));
        helper->free_list->push_back(expr);
        this->lambda_expression_->cuda_allocate(expr, helper);
    }

    const node_t result(this,
        cuda_to_array(&invariant_lst, helper->free_list),
        cuda_to_array(&edge_lst, helper->free_list), expr);
    cudaMemcpy(pointer, &result, sizeof(node_t), cudaMemcpyHostToDevice);
}


CPU GPU bool node_t::is_branch_point() const
{
    return this->is_branch_point_;
}

CPU GPU bool node_t::max_time_progression(simulator_state* state, double* out_max_progression) const
{
    if(this->invariants_.size() <= 0) return false;

    double node_max = -1.0;
    for (int i = 0; i < this->invariants_.size(); ++i)
    {
        double local_max = -1.0;
        if(this->invariants_.get(i)->check_max_time_progression(state, &local_max))
        {
            if(node_max < 0) node_max = local_max;
            else node_max = node_max < local_max ? node_max : local_max;
        }
    }

    (*out_max_progression) = node_max;
    return node_max >= 0; //initial value is negative. Not allowed to set negative value
}
