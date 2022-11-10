// ReSharper disable CppClangTidyBugproneSizeofExpression
#include "stochastic_model_t.h"

stochastic_model_t::stochastic_model_t(
    const array_t<node_t*> models,
    const array_t<clock_variable> timers,
    const array_t<clock_variable> variables,
    const unsigned channel_count)
{
    this->models_ = models;
    this->timers_ = timers;
    this->variables_ = variables;
    this->channel_count_ = channel_count;
}

CPU GPU unsigned stochastic_model_t::get_channel_count() const
{
    return this->channel_count_;
}

unsigned stochastic_model_t::get_variable_count() const
{
    return this->variables_.size();
}

unsigned stochastic_model_t::get_timer_count() const
{
    return this->timers_.size();
}

unsigned stochastic_model_t::get_models_count() const
{
    return this->models_.size();
}

void stochastic_model_t::cuda_allocate(stochastic_model_t* device, allocation_helper* helper) const
{
    //allocate models!
    node_t* node_store = nullptr;
    node_t** node_store_p = nullptr;
    node_t** local_node_store_p = static_cast<node_t**>(malloc(sizeof(node_t*)*this->models_.size()));
    helper->allocate(&node_store, sizeof(node_t)*this->models_.size());
    helper->allocate(&node_store_p, sizeof(node_t*)*this->models_.size());
    const array_t<node_t*> node_arr = array_t<node_t*>(node_store_p, this->models_.size());

    for (int i = 0; i < this->models_.size(); ++i)
    {
        this->models_.get(i)->cuda_allocate(&node_store[i], helper);
        local_node_store_p[i] = &node_store[i];
    }
    cudaMemcpy(node_store_p, local_node_store_p, sizeof(node_t*)*this->models_.size(), cudaMemcpyHostToDevice);
    free(local_node_store_p);

    //allocate clocks
    clock_variable* clock_store = nullptr;
    helper->allocate(&clock_store, sizeof(clock_variable)*this->timers_.size());
    const array_t<clock_variable> clock_arr = array_t<clock_variable>(clock_store, this->timers_.size());
    for (int i = 0; i < this->timers_.size(); ++i)
    {
        this->timers_.at(i)->cuda_allocate(&clock_store[i], helper);
    }

    //allocate clocks
    clock_variable* variable_store = nullptr;
    helper->allocate(&variable_store, sizeof(clock_variable)*this->variables_.size());
    const array_t<clock_variable> variable_arr = array_t<clock_variable>(variable_store, this->variables_.size());
    for (int i = 0; i < this->variables_.size(); ++i)
    {
        this->variables_.at(i)->cuda_allocate(&variable_store[i], helper);
    }
    
    const stochastic_model_t result = stochastic_model_t(node_arr, clock_arr, variable_arr, this->channel_count_);
    cudaMemcpy(device, &result, sizeof(stochastic_model_t), cudaMemcpyHostToDevice);
}

void stochastic_model_t::accept(visitor* v) const
{
    //visit timers
    for (int i = 0; i < this->timers_.size(); ++i)
    {
        v->visit(this->timers_.at(i));
    }

    //visit variables
    for (int i = 0; i < this->variables_.size(); ++i)
    {
        v->visit(this->variables_.at(i));
    }
    
    //visit models
    for (int i = 0; i < this->models_.size(); ++i)
    {
        v->visit(this->models_.get(i));
    }
}

// ReSharper disable once CppMemberFunctionMayBeStatic
void stochastic_model_t::pretty_print() const
{
    //TODO fix plz :)
    return;
} 
