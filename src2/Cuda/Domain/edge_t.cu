#include "edge_t.h"


edge_t::edge_t(edge_t* source, node_t* dest, constraint_t* guard, array_t<update_t*> updates)
{
    this->id_ = source->id_;
    this->dest_ = dest;
    this->weight_ = source->weight_;
    this->updates_ = updates;
    this->guard_ = guard;
}

edge_t::edge_t(const int id, const float weight, node_t* dest, constraint_t* guard)
{
    this->id_ = id;
    this->dest_ = dest;
    this->weight_ = weight;
    this->updates_ = array_t<update_t*>(0);
    this->guard_ = guard;
}

CPU GPU float edge_t::get_weight() const
{
    return this->weight_;
}

GPU CPU node_t* edge_t::get_dest() const
{
    return this->dest_;
}

void edge_t::set_updates(std::list<update_t*>* updates)
{
    this->updates_ = to_array(updates);
}

GPU bool edge_t::evaluate_constraints(const lend_array<clock_timer_t>* timers) const
{
    if(this->guard_ == nullptr) return true;
    return this->guard_->evaluate(timers);
}

void edge_t::accept(visitor* v)
{
    printf("    ");
    v->visit(this->guard_);
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        update_t* temp = *updates_.at(i);
        v->visit(*&*&*&*&*&*&*&*&*&*&temp);
    }
}


int edge_t::get_id() const
{
    return this->id_;
}

void edge_t::cuda_allocate(edge_t** pointer, std::list<void*>* free_list)
{
    cudaMalloc(pointer, sizeof(edge_t));
    free_list->push_back(*pointer);

    node_t* node_p = nullptr;
    this->dest_->cuda_allocate(&node_p, free_list);

    constraint_t* guard_p = nullptr;
    this->guard_->cuda_allocate(&guard_p, free_list);

    std::list<update_t*> update_pointers;
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        update_t* update = nullptr;
        update_t* temp = *updates_.at(i);
        temp->cuda_allocate(&update, free_list);
        update_pointers.push_back(update);
    }
    edge_t result(this, node_p, guard_p, cuda_to_array(&update_pointers, free_list));
    cudaMemcpy(*pointer, &result, sizeof(edge_t), cudaMemcpyHostToDevice);
}


GPU void edge_t::execute_updates(const lend_array<clock_timer_t>* timers)
{
    for (int i = 0; i < this->updates_.size(); ++i)
    {
        update_t* temp = *updates_.at(i);
        temp->update_timer(timers);
    }
}
