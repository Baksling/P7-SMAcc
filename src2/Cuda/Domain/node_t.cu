#include "node_t.h"
#include "simulator_state.h"

node_t::node_t(const node_t* source, const array_t<constraint_t>& invariant, const array_t<edge_t>& edges, expression* lambda)
{
    this->id_ = source->id_;
    this->is_branch_point_ = source->is_branch_point_;
    this->invariants_ = invariant;
    this->is_goal_ = source->is_goal_;
    this->edges_ = edges;
    this->lambda_expression_ = lambda;
}


node_t::node_t(const int id, const array_t<constraint_t>& invariants,
    const bool is_branch_point, const bool is_goal, expression* lambda)
{
    //if no lambda supplied, defaults to 1, as to imitate UPPAAL.
    if(lambda == nullptr)
    {
        lambda = expression::literal_expression(1);
    }
    
    this->id_ = id;
    this->is_goal_ = is_goal;
    this->invariants_ = invariants;
    this->lambda_expression_ = lambda;
    this->is_branch_point_ = is_branch_point;
    this->edges_ = array_t<edge_t>(0);
}


int node_t::get_id() const
{
    return this->id_;
}

GPU CPU double node_t::get_lambda(simulator_state* state) const
{
    if(this->lambda_expression_ != nullptr)
        return this->lambda_expression_->evaluate(state);

    return 0.0; //illegal lambda value.
}

void node_t::set_edges(std::list<edge_t>* list)
{
    this->edges_ = to_array(list);
}

CPU GPU lend_array<edge_t> node_t::get_edges()
{
    return lend_array<edge_t>(&this->edges_);
}

CPU GPU bool node_t::is_goal_node() const
{
    return this->is_goal_;
}

CPU GPU bool node_t::evaluate_invariants(simulator_state* state) const
{
    for (int i = 0; i < this->invariants_.size(); ++i)
    {
        if(!this->invariants_.at(i)->evaluate(state))
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
        v->visit(this->invariants_.at(i));
    }

    //visit edges
    for (int i = 0; i < this->edges_.size(); ++i)
    {
        v->visit(this->edges_.at(i));
    }

    //visit edge destinations
    for (int i = 0; i < this->edges_.size(); ++i)
    {
        v->visit(this->edges_.at(i)->get_dest());
    }
}

void node_t::pretty_print() const
{
    printf("\nNode id: %3d | Is branch: %d | Is goal: %d \n",
        this->id_,
        this->is_branch_point_,
        this->is_goal_);
}

void node_t::cuda_allocate(node_t* pointer, allocation_helper* helper)
{
    if(helper->node_map.count(this) == 1) return;

    //add current node to circular reference resolver
    //pointer is the cuda location this node will be stored at
    helper->node_map.insert( std::pair<node_t*, node_t*>(this, pointer) );
    
    std::list<edge_t*> edge_lst;
    edge_t* edge_p = nullptr;
    helper->allocate_cuda(&edge_p, sizeof(edge_t)*this->edges_.size());
    const array_t<edge_t> edge_arr = array_t<edge_t>(edge_p, this->edges_.size());
    for (int i = 0; i < this->edges_.size(); ++i)
    {
        this->edges_.at(i)->cuda_allocate(&edge_p[i], helper);
    }

    constraint_t* invariant_p = nullptr;
    helper->allocate_cuda(&invariant_p, sizeof(constraint_t)*this->invariants_.size());
    const array_t<constraint_t> constraint_arr = array_t<constraint_t>(invariant_p, this->invariants_.size());
    for (int i = 0; i < this->invariants_.size(); ++i)
    {
        this->invariants_.at(i)->cuda_allocate(&invariant_p[i], helper);
    }

    expression* expr = nullptr;
    if(this->lambda_expression_ != nullptr)
    {
        helper->allocate_cuda(&expr, sizeof(expression));
        this->lambda_expression_->cuda_allocate(expr, helper);
    }

    const node_t result(this,
        constraint_arr,
        edge_arr,
        expr);
    cudaMemcpy(pointer, &result, sizeof(node_t), cudaMemcpyHostToDevice);
}


CPU GPU bool node_t::is_branch_point() const
{
    return this->is_branch_point_;
}

bool node_t::is_progressible() const
{
    for (int j = 0; j < this->edges_.size(); ++j)
    {
        if (!this->edges_.at(j)->is_listener())
        {
            return true;
        }
    }

    return false;
}

CPU GPU bool node_t::max_time_progression(simulator_state* state, double* out_max_progression) const
{
    if(this->invariants_.size() <= 0) return false;

    double node_max = -1.0;
    for (int i = 0; i < this->invariants_.size(); ++i)
    {
        double local_max = -1.0;
        if(this->invariants_.at(i)->check_max_time_progression(state, &local_max))
        {
            if(node_max < 0) node_max = local_max;
            else node_max = node_max < local_max ? node_max : local_max;
        }
    }

    (*out_max_progression) = node_max;
    return node_max >= 0; //initial value is negative. Not allowed to set negative value
}
