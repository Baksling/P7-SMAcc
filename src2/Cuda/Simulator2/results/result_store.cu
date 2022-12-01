#include "result_store.h"

result_pointers::result_pointers(const bool owns_pointers, void* free_p, sim_metadata* results, int* nodes,
    double* variables): owns_pointers_(owns_pointers), source_p_(free_p)
{
    this->meta_results = results;
    this->nodes = nodes;
    this->variables = variables;
}

void result_pointers::free_internals() const
{
    if(!owns_pointers_) return;
    free(this->source_p_);
}

size_t result_store::total_data_size() const
{
    return sizeof(sim_metadata) * this->simulations_
         + sizeof(double)       * this->simulations_ * this->variables_count_
         + sizeof(int)          * this->simulations_ * this->models_count_;
}

result_store::result_store(const unsigned total_sim, const unsigned variables, const unsigned network_size, memory_allocator* helper)
{
    this->is_cuda_ = helper->use_cuda;
    this->simulations_ = static_cast<int>(total_sim);
    this->models_count_ = static_cast<int>(network_size);
    this->variables_count_ = static_cast<int>(variables);

    const size_t total_data = this->total_data_size();

    void* store;
    CUDA_CHECK(helper->allocate(&store, total_data));

    this->metadata_p_ = static_cast<sim_metadata*>(store);
    store = static_cast<void*>(&this->metadata_p_[this->simulations_]);
        
    this->variable_p_ = static_cast<double*>(store);
    store = static_cast<void*>(&this->variable_p_[(this->variables_count_ * this->simulations_)]);
        
    this->node_p_ = static_cast<int*>(store);
}

result_pointers result_store::load_results() const
{
    if(!this->is_cuda_) return result_pointers{
        false,
        nullptr,
        this->metadata_p_,
        this->node_p_,
        this->variable_p_
    };

    const size_t size = this->total_data_size();
    const void* source = static_cast<void*>(this->metadata_p_); //this is the source of the array. nodes and variables are just offsets from here
    void* init_store = malloc(size);
    void* store = init_store;
    CUDA_CHECK(cudaMemcpy(store, source , size, cudaMemcpyDeviceToHost));

    sim_metadata* meta = static_cast<sim_metadata*>(store);
    store = static_cast<void*>(&meta[this->simulations_]);
        
    double* vars = static_cast<double*>(store);
    store = static_cast<void*>(&vars[static_cast<int>(this->variables_count_ * this->simulations_)]);
        
    int* nodes = static_cast<int*>(store);
    return result_pointers{
        true,
        init_store,
        meta,
        nodes,
        vars
    };
}
