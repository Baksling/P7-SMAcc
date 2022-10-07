#include "stochastic_model_t.h"

stochastic_model_t::stochastic_model_t(node_t* start_node, const array_t<clock_timer_t*> timers)
{
    this->start_node_ = start_node;
    this->timers_ = timers;
}
void stochastic_model_t::accept(visitor* v)
{
    v->visit(this->start_node_);
    for (int i = 0; i < this->timers_.size(); ++i)
    {
        clock_timer_t* temp = *this->timers_.at(i);
        v->visit(temp);
    }
}

GPU array_t<clock_timer_t> stochastic_model_t::create_internal_timers()
{
    const int size = this->timers_.size();
    clock_timer_t* internal_timers_arr =
        static_cast<clock_timer_t*>(malloc(sizeof(clock_timer_t) * size));
    
    
    for (int i = 0; i < size; i++)
    {
        internal_timers_arr[i] = this->timers_.get(i)->duplicate();
    }

    const array_t<clock_timer_t> internal_timers{ internal_timers_arr, size};
    return internal_timers;
}

GPU void stochastic_model_t::reset_timers(array_t<clock_timer_t>* active_timers)
{
    if(active_timers->size() != this->timers_.size())
    {
        printf("Timers mismatch!!");
        return;
    }
    
    for (int i = 0; i < active_timers->size(); i++)
    {
        clock_timer_t* timer = active_timers->at(i);
        const clock_timer_t* temp = *this->timers_.at(i);
        timer->set_time(temp->get_time());
    }
}

GPU node_t* stochastic_model_t::get_start_node() const
{
    return this->start_node_;
}

void stochastic_model_t::cuda_allocate(stochastic_model_t** pointer, const allocation_helper* helper)
{
    cudaMalloc(pointer, sizeof(stochastic_model_t));
    helper->free_list->push_back(*pointer);

    //allocate start node
    node_t* node_p = nullptr;
    this->start_node_->cuda_allocate(&node_p, helper);

    std::list<clock_timer_t*> clocks_lst;
    for (int i = 0; i < timers_.size(); ++i)
    {
        clock_timer_t* timer_p = nullptr;
        this->timers_.get(i)->cuda_allocate(&timer_p, helper);
        clocks_lst.push_back(timer_p);
    }
    
    const stochastic_model_t result(node_p, cuda_to_array(&clocks_lst, helper->free_list));
    cudaMemcpy(*pointer, &result, sizeof(stochastic_model_t), cudaMemcpyHostToDevice);
}
