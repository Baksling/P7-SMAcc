#include "stochastic_model_t.h"

stochastic_model_t::stochastic_model_t(node_t* start_node, const array_t<clock_variable*> timers,
    const array_t<clock_variable*> variables)
{
    this->start_node_ = start_node;
    this->timers_ = timers;
    this->variables_ = variables;
}

void stochastic_model_t::accept(visitor* v) const
{
    v->visit(this->start_node_);
    for (int i = 0; i < this->timers_.size(); ++i)
    {
        clock_variable* temp = *this->timers_.at(i);
        v->visit(temp);
    }
}

void stochastic_model_t::pretty_print() const
{
    printf("Model start\n");
}

CPU GPU array_t<clock_variable> stochastic_model_t::create_internal_timers() const
{
    const int size = this->timers_.size();
    if(size == 0) return array_t<clock_variable>(0);

    clock_variable* internal_timers_arr =
        static_cast<clock_variable*>(malloc(sizeof(clock_variable) * size));
    
    for (int i = 0; i < size; i++)
    {
        internal_timers_arr[i] = this->timers_.get(i)->duplicate();
    }

    const array_t<clock_variable> internal_timers{ internal_timers_arr, size};
    return internal_timers;
}

array_t<clock_variable> stochastic_model_t::create_internal_variables() const
{
    const int size = this->variables_.size();
    if(size == 0) return array_t<clock_variable>(0);
    clock_variable* internal_variable_arr =
        static_cast<clock_variable*>(malloc(sizeof(clock_variable) * size));
    
    for (int i = 0; i < size; i++)
    {
        internal_variable_arr[i] = this->variables_.get(i)->duplicate();
    }

    const array_t<clock_variable> internal_timers{ internal_variable_arr, size};
    return internal_timers;
}

CPU GPU void stochastic_model_t::reset_timers(const array_t<clock_variable>* active_timers, const array_t<clock_variable>* active_variables) const
{
    if(active_timers->size() != this->timers_.size())
    {
        printf("Timers mismatch!!");
        return;
    }

    if(active_variables->size() != this->variables_.size())
    {
        printf("Variable mismatch!!");
        return;
    }
    
    for (int i = 0; i < active_timers->size(); i++)
    {
        clock_variable* timer = active_timers->at(i);
        const clock_variable* initial = *this->timers_.at(i);
        timer->set_time(initial->get_time());
    }

    for (int i = 0; i < active_variables->size(); i++)
    {
        clock_variable* variable = active_variables->at(i);
        const clock_variable* initial = *this->variables_.at(i);
        variable->set_time(initial->get_time());
    }
}

GPU node_t* stochastic_model_t::get_start_node() const
{
    return this->start_node_;
}

void stochastic_model_t::cuda_allocate(stochastic_model_t** pointer, const allocation_helper* helper) const
{
    cudaMalloc(pointer, sizeof(stochastic_model_t));
    helper->free_list->push_back(*pointer);

    //allocate start node
    node_t* node_p = nullptr;
    this->start_node_->cuda_allocate(&node_p, helper);

    std::list<clock_variable*> clocks_lst;
    for (int i = 0; i < this->timers_.size(); ++i)
    {
        clock_variable* timer_p = nullptr;
        cudaMalloc(&timer_p, sizeof(clock_variable));
        clocks_lst.push_back(timer_p);
        this->timers_.get(i)->cuda_allocate(timer_p, helper);
    }

    std::list<clock_variable*> variable_lst;
    for (int i = 0; i < this->variables_.size(); ++i)
    {
        clock_variable* variable_p = nullptr;
        cudaMalloc(&variable_p, sizeof(clock_variable));
        variable_lst.push_back(variable_p);
        this->variables_.get(i)->cuda_allocate(variable_p, helper);
    }
    
    const stochastic_model_t result(node_p,
        cuda_to_array(&clocks_lst, helper->free_list),
        cuda_to_array(&variable_lst, helper->free_list));
    
    cudaMemcpy(*pointer, &result, sizeof(stochastic_model_t), cudaMemcpyHostToDevice);
}
