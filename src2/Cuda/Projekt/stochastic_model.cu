#include "stochastic_model.h"
#include <assert.h>
#include <ctime>

using namespace std;

stochastic_model::stochastic_model(uneven_list<edge_d>* node_to_edge, uneven_list<guard_d>* node_to_invariant,
                                   uneven_list<guard_d>* edge_to_guard, uneven_list<update_d>* edge_to_update,
                                   timer_d* timers, const int timer_count)
{
    this->timer_count_ = timer_count;
    this->timers_ = timers;
    this->node_to_edge_ = node_to_edge;
    this->node_to_invariant_ = node_to_invariant;
    this->edge_to_guard_ = edge_to_guard;
    this->edge_to_update_ = edge_to_update;
}


GPU array_info<edge_d> stochastic_model::get_node_edges(const int node_id) const
{
    return this->node_to_edge_->get_index(node_id);
}

GPU array_info<guard_d> stochastic_model::get_node_invariants(const int node_id) const
{
    return this->node_to_invariant_->get_index(node_id);
}

GPU array_info<guard_d> stochastic_model::get_edge_guards(const int edge_id) const
{
    return this->edge_to_guard_->get_index(edge_id);

}

GPU array_info<update_d> stochastic_model::get_updates(const int edge_id) const
{
    return this->edge_to_update_->get_index(edge_id);
}


GPU void stochastic_model::traverse_edge_update(const int edge_id, const array_info<timer_d>* local_timers) const
{
    const array_info<update_d> updates = this->get_updates(edge_id);

    for (int i = 0; i < updates.size; ++i)
    {
        update_d* update = &updates.arr[i];
        const int timer_id = update->get_timer_id();
        timer_d* timer = &local_timers->arr[timer_id];

        timer->set_time(update->get_value());
    }
    
    updates.free_arr();
}

GPU int stochastic_model::get_start_node() const
{
    return 0;
}

GPU bool stochastic_model::is_goal_node(int node_id) const
{
    return node_id == 2;
}

GPU array_info<timer_d> stochastic_model::copy_timers() const
{
    const int size = this->timer_count_;
    timer_d* internal_timers_arr = static_cast<timer_d*>(malloc(sizeof(timer_d) * size));
    const array_info<timer_d> internal_timers{ internal_timers_arr, size};
    
    for (int i = 0; i < internal_timers.size; i++)
    {
        internal_timers.arr[i] = this->timers_[i].copy();
    }
    
    return internal_timers;
}

GPU void stochastic_model::reset_timers(const array_info<timer_d>* timers) const
{
    assert(timers->size == this->timer_count_);
    for (int i = 0; i < timers->size; i++)
    {
        timers->arr[i].set_time(this->timers_[i].get_value());
    }
    
}

void stochastic_model::cuda_allocate(stochastic_model** p, list<void*>* free_list) const
{
    //move internal lists to cuda    
    uneven_list<edge_d>* node_to_edge_d = nullptr;
    this->node_to_edge_->cuda_allocate(&node_to_edge_d, free_list);
    
    uneven_list<guard_d>* node_to_invariant_d = nullptr;
    this->node_to_invariant_->cuda_allocate(&node_to_invariant_d, free_list);

    uneven_list<guard_d>* edge_to_guard_d = nullptr;
    this->edge_to_guard_->cuda_allocate(&edge_to_guard_d, free_list);

    uneven_list<update_d>* edge_to_update_d = nullptr;
    this->edge_to_update_->cuda_allocate(&edge_to_update_d, free_list);

    //move timers to cuda
    timer_d* timers_d = nullptr;
    cudaMalloc(&timers_d, sizeof(timer_d)*this->timer_count_);
    free_list->push_back(timers_d);
    cudaMemcpy(timers_d, this->timers_, sizeof(timer_d)*this->timer_count_, cudaMemcpyHostToDevice);

    //create model with cuda pointers
    const stochastic_model model = stochastic_model(
        node_to_edge_d, node_to_invariant_d,
        edge_to_guard_d, edge_to_update_d,
        timers_d, this->timer_count_);

    //move model with cuda pointers to device. Add to free list.
    cudaMalloc(p, sizeof(stochastic_model));
    free_list->push_back((*p));
    cudaMemcpy((*p), &model, sizeof(stochastic_model), cudaMemcpyHostToDevice);
}

CPU GPU void stochastic_model::pretty_print() const
{
    for (int node_id = 0; node_id < this->node_to_edge_->get_index_size(); ++node_id)
    {
        printf("Node: %d \n", node_id);
        array_info<edge_d> edges = this->node_to_edge_->get_index(node_id);
        for (int edge_id = 0; edge_id < edges.size; ++edge_id)
        {
            array_info<guard_d> guards = this->edge_to_guard_->get_index(edge_id);
            array_info<update_d> updates = this->edge_to_update_->get_index(edge_id);
            printf("    Edge id: %d, %d -> %d \n", edge_id, node_id, edges.arr[edge_id].get_dest_node());
            printf("        Guards amount: %d \n", guards.size);
            printf("        Updates: %d \n", updates.size);
            for (int update_id = 0; update_id < updates.size; ++update_id)
            {
                printf("            Clock: %d, Value: %f \n", updates.arr[update_id].get_timer_id(), updates.arr[update_id].get_value());
            }
            updates.free_arr();
            guards.free_arr();
        }
        edges.free_arr();
    }
    printf("Clocks %d \n", this->timer_count_);
    for (int clock_id = 0; clock_id < this->timer_count_; ++clock_id)
    {
        printf("    Clock: %d, Value: %f \n", this->timers_[clock_id].get_id(), this->timers_->get_value());       
    }
}
