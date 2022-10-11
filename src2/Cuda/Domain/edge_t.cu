#include "edge_t.h"


edge_t::edge_t(edge_t* source, node_t* dest, const array_t<constraint_t*> guard, const array_t<update_t*> updates)
{
    this->id_ = source->id_;
    this->dest_ = dest;
    this->weight_ = source->weight_;
    this->updates_ = updates;
    this->guards_ = guard;
}

edge_t::edge_t(const int id, const float weight, node_t* dest, const array_t<constraint_t*> guard)
{
    this->id_ = id;
    this->dest_ = dest;
    this->weight_ = weight;
    this->updates_ = array_t<update_t*>(0);
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

void edge_t::set_updates(std::list<update_t*>* updates)
{
    this->updates_ = to_array(updates);
}

CPU GPU bool edge_t::evaluate_constraints(const lend_array<clock_timer_t>* timers) const
{
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        update_t* update = this->updates_.get(i);
        clock_timer_t* clock = timers->at(update->get_timer_id());
        if (clock->get_temp_time() > update->get_timer_value())
        {
            clock->set_temp_time(update->get_timer_value());
        }
    }
    const bool valid_dest = this->dest_->evaluate_invariants(timers);

    for (int i = 0; i < this->updates_.size(); ++i)
    {
        clock_timer_t* clock = timers->at(this->updates_.get(i)->get_timer_id());
        if (clock != nullptr)
            clock->reset_temp_time();
    }
    
    if(!valid_dest) return false;

    for (int i = 0; i < this->guards_.size(); ++i)
    {
        if(!this->guards_.get(i)->evaluate(timers))
            return false;
    }
    
    return true;
}

void edge_t::accept(visitor* v) const
{
    //visit edge guards
    for (int i = 0; i < this->guards_.size(); ++i)
    {
        printf("        ");
        v->visit(this->guards_.get(i));
    }

    //visit edge updates
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        update_t* temp = *updates_.at(i);
        v->visit(*&*&*&*&*&*&*&*&*&*&temp);
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

    std::list<constraint_t*> invariant_lst;
    for (int i = 0; i < this->guards_.size(); ++i)
    {
        constraint_t* invariant_p = nullptr;
        this->guards_.get(i)->cuda_allocate(&invariant_p, helper);
        invariant_lst.push_back(invariant_p);
    }
    
    std::list<update_t*> updates;
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        update_t* update_p = nullptr;
        this->updates_.get(i)->cuda_allocate(&update_p, helper);
        updates.push_back(update_p);
    }
    
    const edge_t result(this, node_p,
        cuda_to_array(&invariant_lst, helper->free_list), cuda_to_array(&updates, helper->free_list));
    cudaMemcpy(*pointer, &result, sizeof(edge_t), cudaMemcpyHostToDevice);
}


void edge_t::cuda_allocate_2(edge_t* cuda_p, const allocation_helper* helper)
{
    return;
}


CPU GPU void edge_t::execute_updates(const lend_array<clock_timer_t>* timers) const
{
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        this->updates_.get(i)->apply_update(timers);
    }
}
