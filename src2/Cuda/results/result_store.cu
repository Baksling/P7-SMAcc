﻿#include "result_store.h"
#include <iostream>
#include <string>

result_pointers::result_pointers(const bool owns_pointers,
    void* free_p,
    node_results* nodes,
    variable_result* variables,
    const int threads,
    const unsigned total_simulations): owns_pointers_(owns_pointers), source_p_(free_p)
{
    this->nodes = nodes;
    this->variables = variables;
    this->total_simulations = total_simulations;
    this->threads = threads;
}

void result_pointers::free_internals() const
{
    if(!owns_pointers_) return;
    free(this->source_p_);
}

size_t result_store::total_data_size() const
{
    return sizeof(node_results)    * this->node_count_ * this->thread_count_
         + sizeof(variable_result) * this->variables_count_ * this->thread_count_;
}


result_store::result_store(const unsigned total_sim,
    const unsigned variables,
    const unsigned node_count,
    const int thread_count, memory_allocator* helper)
{
    this->is_cuda_ = helper->use_cuda;
    this->simulations_ = total_sim;
    this->node_count_ = node_count;
    this->variables_count_ = variables;
    this->thread_count_ = thread_count;

    const size_t total_data = this->total_data_size();

    void* store;
    CUDA_CHECK(helper->allocate(&store, total_data));
    
    this->node_p_ = static_cast<node_results*>(store);
    store = static_cast<void*>(&this->node_p_[this->node_count_ * this->thread_count_]);
        
    this->variable_p_ = static_cast<variable_result*>(store);
}

result_pointers result_store::load_results() const
{
    if(!this->is_cuda_) return result_pointers{
        false,
        nullptr,
        this->node_p_,
        this->variable_p_,
        this->thread_count_,
        this->simulations_
    };

    const size_t size = this->total_data_size();
    const void* source = static_cast<void*>(this->node_p_); //this is the source of the array. nodes and variables are just offsets from here
    void* init_store = malloc(size);
    void* store = init_store;
    CUDA_CHECK(cudaMemcpy(store, source , size, cudaMemcpyDeviceToHost));

    node_results* nodes = static_cast<node_results*>(store);
    const int offset = static_cast<int>(this->node_count_) * this->thread_count_;
    store = static_cast<void*>(&nodes[offset]);
        
    variable_result* vars = static_cast<variable_result*>(store);
    return result_pointers{
        true,
        init_store,
        nodes,
        vars,
        this->thread_count_,
        this->simulations_
    };
}
