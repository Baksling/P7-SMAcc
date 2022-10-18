﻿#include "edge_t.h"
#include "node_t.h"


edge_t::edge_t(edge_t* source, node_t* dest, const array_t<constraint_t*> guard, const array_t<update_t*> updates)
{
    this->id_ = source->id_;
    this->dest_ = dest;
    this->weight_ = source->weight_;
    this->updates_ = updates;
    this->guards_ = guard;
}

edge_t::edge_t(const int id, const float weight, node_t* dest, const array_t<constraint_t*> guard, array_t<update_t*> updates)
{
    this->id_ = id;
    this->dest_ = dest;
    this->weight_ = weight;
    this->updates_ = updates;
    this->guards_ = guard;
}

CPU GPU float edge_t::get_weight() const
{
    return this->weight_;
}

CPU GPU node_t* edge_t::get_dest() const
{
    return this->dest_;
}

CPU GPU bool edge_t::evaluate_constraints(simulator_state* state) const
{
    //Evaluate guards
    for (int i = 0; i < this->guards_.size(); ++i)
    {
        if(!this->guards_.get(i)->evaluate(&state->timers))
            return false;
    }

    //Evaluate destination node
    //Only if guards pass
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        const update_t* update = this->updates_.get(i);
        update->apply_temp_update(state);
    }
    //check destination node using temporary update values
    const bool valid_dest = this->dest_->evaluate_invariants(state);

    //reset updates
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        this->updates_.get(i)->reset_temp_update(state);
    }

    //reset updates first
    return valid_dest;
}

CPU GPU void edge_t::execute_updates(simulator_state* state) const
{
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        this->updates_.get(i)->apply_update(state);
    }
}

void edge_t::accept(visitor* v) const
{
    //visit edge guards
    for (int i = 0; i < this->guards_.size(); ++i)
    {
        printf("        "); //TODO wtf is this?
        v->visit(this->guards_.get(i));
    }

    //visit edge updates
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        update_t* temp = *updates_.at(i);
        v->visit(temp);
    }

    //dont visit destination. Handled by node itself.
}


int edge_t::get_id() const
{
    return this->id_;
}

void edge_t::cuda_allocate(edge_t** pointer, const allocation_helper* helper)
{
    cudaMalloc(pointer, sizeof(edge_t));
    helper->free_list->push_back(*pointer);

    //allocate node
    node_t* node_p = nullptr;
    if (helper->node_map->count(this->dest_) == 1)
    {
        node_p = (*helper->node_map)[this->dest_];
    }
    else
    {
        this->dest_->cuda_allocate(&node_p, helper); //linear node
    }

    std::list<constraint_t*> guard_lst;
    for (int i = 0; i < this->guards_.size(); ++i)
    {
        constraint_t* invariant_p = nullptr;
        this->guards_.get(i)->cuda_allocate(&invariant_p, helper);
        guard_lst.push_back(invariant_p);
    }
    
    std::list<update_t*> updates;
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        update_t* update_p = nullptr;
        cudaMalloc(&update_p, sizeof(update_t));
        helper->free_list->push_back(update_p);
        
        this->updates_.get(i)->cuda_allocate(update_p, helper);
        updates.push_back(update_p);
    }
    
    const edge_t result(this, node_p,
        cuda_to_array(&guard_lst, helper->free_list), cuda_to_array(&updates, helper->free_list));
    cudaMemcpy(*pointer, &result, sizeof(edge_t), cudaMemcpyHostToDevice);
}


void edge_t::cuda_allocate_2(edge_t* cuda_p, const allocation_helper* helper)
{
    return;
}

int edge_t::get_updates_size() const
{
    return this->updates_.size();
}
