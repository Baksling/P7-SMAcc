#include "edge_t.h"
#include "node_t.h"

edge_t::edge_t(
    const int id,
    expression* weight_expression,
    node_t* dest,
    const array_t<constraint_t>& guards,
    const array_t<update_t>& updates,
    const edge_channel channel
    )
{
    this->id_ = id;
    this->dest_ = dest;
    this->weight_expression_ = weight_expression;
    this->updates_ = updates;
    this->guards_ = guards;
    this->channel_ = channel;
}

CPU GPU double edge_t::get_weight(simulator_state* state) const
{
    return this->weight_expression_->evaluate(state);
}

CPU GPU unsigned edge_t::get_channel() const
{
    return this->channel_.channel_id;
}

bool edge_t::is_listener() const
{
    return this->channel_.is_listener && this->channel_.channel_id != NO_CHANNEL;
}

int edge_t::get_id() const
{
    return this->id_;
}


node_t* edge_t::get_dest() const
{
    return this->dest_;
}

CPU GPU bool edge_t::evaluate_constraints(simulator_state* state) const
{
    //Evaluate guards
    for (int i = 0; i < this->guards_.size(); ++i)
    {
        if(!this->guards_.at(i)->evaluate(state))
            return false;
    }

    //Evaluate destination node
    //Only if guards pass
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        const update_t* update = this->updates_.at(i);
        update->apply_temp_update(state);
    }
    //check destination node using temporary update values
    const bool valid_dest = this->dest_->evaluate_invariants(state);

    //reset updates
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        this->updates_.at(i)->reset_temp_update(state);
    }

    //reset updates first
    return valid_dest;
}

CPU GPU void edge_t::execute_updates(simulator_state* state) const
{
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        this->updates_.at(i)->apply_update(state);
    }
}

void edge_t::accept(visitor* v) const
{
    //visit weight
    v->visit(this->weight_expression_);
    
    //visit edge guards
    for (int i = 0; i < this->guards_.size(); ++i)
    {
        v->visit(this->guards_.at(i));
    }

    //visit edge updates
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        update_t* temp = updates_.at(i);
        v->visit(temp);
    }

    v->visit(this->weight_expression_);
}

void edge_t:: pretty_print() const
{
    printf("Edge id: %3d | Weight expression: %s | Dest node: %3d | Channel Id: %3s | Is Listener: %s\n",
        this->id_,
        this->weight_expression_->to_string().c_str(),
        this->dest_->get_id(),
        (this->get_channel() == NO_CHANNEL ? "N/A" : std::to_string(this->get_channel())).c_str(),
        this->is_listener() ? "True" : "False");
}

void edge_t::cuda_allocate(edge_t* pointer, allocation_helper* helper) const
{
    //allocate node
    node_t* node_p = nullptr;
    if (helper->node_map.count(this->dest_) == 1)
    {
        node_p = helper->node_map[this->dest_];
    }
    else
    {   //allocate node if its not already allocated
        //The node's cuda_allocate method is responsible for adding it to the circular reference resolver
        helper->allocate_cuda(&node_p, sizeof(node_t));
        this->dest_->cuda_allocate(node_p, helper); //linear node
    }

    constraint_t* guard_p = nullptr;
    helper->allocate_cuda(&guard_p, sizeof(constraint_t)*this->guards_.size());
    for (int i = 0; i < this->guards_.size(); ++i)
    {
        this->guards_.at(i)->cuda_allocate(&guard_p[i], helper);
    }
    
    update_t* updates_d = nullptr;
    helper->allocate_cuda(&updates_d, sizeof(update_t)*this->updates_.size());
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        this->updates_.at(i)->cuda_allocate(&updates_d[i], helper);
    }

    expression* weight_p = nullptr;
    helper->allocate_cuda(&weight_p, sizeof(expression));
    this->weight_expression_->cuda_allocate(weight_p, helper);
    
    
    const edge_t result(
        this->id_,
        weight_p,
        node_p,
        array_t<constraint_t>(guard_p, this->guards_.size()),
        array_t<update_t>(updates_d, this->updates_.size()),
        this->channel_);
    
    cudaMemcpy(pointer, &result, sizeof(edge_t), cudaMemcpyHostToDevice);
}


int edge_t::get_updates_size() const
{
    return this->updates_.size();
}
