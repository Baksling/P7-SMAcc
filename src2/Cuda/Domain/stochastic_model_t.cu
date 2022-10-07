#include "stochastic_model_t.h"

stochastic_model_t::stochastic_model_t(node_t* start_node, array_t<clock_timer_t*> timers)
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
    clock_timer_t* internal_timers_arr = static_cast<clock_timer_t*>(malloc(sizeof(clock_timer_t) * size));
    
    
    for (int i = 0; i < size; i++)
    {
        clock_timer_t* temp = *this->timers_.at(i); 
        internal_timers_arr[i] = temp->duplicate();
    }

    const array_t<clock_timer_t> internal_timers{ internal_timers_arr, size};
    return internal_timers;
}

GPU void stochastic_model_t::reset_timers(array_t<clock_timer_t>* active_timers)
{
    if(active_timers->size() != this->timers_.size())
    {
        printf("Timers mitchmatch!!");
        return;
    }
    
    for (int i = 0; i < active_timers->size(); i++)
    {
        clock_timer_t* timer = active_timers->at(i);
        clock_timer_t* temp = *this->timers_.at(i);
        timer->set_time(temp->get_time());
    }
}

GPU node_t* stochastic_model_t::get_start_node() const
{
    return this->start_node_;
}

void stochastic_model_t::cuda_allocate(stochastic_model_t** pointer, std::list<void*>* free_list)
{
    cudaMalloc(pointer, sizeof(stochastic_model_t));
    free_list->push_back(*pointer);
    node_t* node_p = nullptr;
    this->start_node_->cuda_allocate(&node_p, free_list);
    std::list<clock_timer_t*> timer_pointer;
    for (int i = 0; i < timers_.size(); ++i)
    {
        clock_timer_t* timer_p = nullptr;
        clock_timer_t* temp = *timers_.at(i);
        temp->cuda_allocate(&timer_p, free_list);
        timer_pointer.push_back(timer_p);
    }
    stochastic_model_t result(node_p, cuda_to_array(&timer_pointer, free_list));
    cudaMemcpy(*pointer, &result, sizeof(stochastic_model_t), cudaMemcpyHostToDevice);
}
