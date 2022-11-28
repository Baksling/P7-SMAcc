#pragma once
#include "memory_allocator.h"
#include "sim_config.cu"
#include "Domain.cu"

#define HAS_HIT_MAX_STEPS(x) ((x) >= 0)

struct state;

struct sim_metadata
{
    unsigned int steps;
    double global_time;
};

struct result_pointers
{
private:
    const bool owns_pointers_;
    void* source_p;
public:
    explicit result_pointers(const bool owns_pointers, void* free_p, sim_metadata* results, int* nodes, double* variables)
        : owns_pointers_(owns_pointers), source_p(free_p)
    {
        this->meta_results = results;
        this->nodes = nodes;
        this->variables = variables;
    }

    sim_metadata* meta_results = nullptr;
    int* nodes = nullptr;
    double* variables = nullptr;

    void free_internals() const
    {
        if(!owns_pointers_) return;
        free(this->source_p);
    }
};

class result_store
{
    friend struct state;
private:
    bool is_cuda_;
    
    int simulations_;
    int models_count_;
    int variables_count_;
    
    sim_metadata* metadata_p_ = nullptr;
    double* variable_p_ = nullptr;
    int* node_p_ = nullptr;
public:
    explicit result_store(const sim_config& config, memory_allocator* helper)
    {
        this->is_cuda_ = helper->use_cuda;
        this->simulations_ = static_cast<int>(config.blocks * config.threads * config.simulation_amount);
        this->models_count_ = static_cast<int>(config.network_size);
        this->variables_count_ = static_cast<int>(config.tracked_variable_count);

        const size_t total_data = this->total_data_size();

        void* store;
        helper->allocate(&store, total_data);

        this->metadata_p_ = static_cast<sim_metadata*>(store);
        store = static_cast<void*>(&this->metadata_p_[this->simulations_]);
        
        this->variable_p_ = static_cast<double*>(store);
        store = static_cast<void*>(&this->variable_p_[(this->variables_count_ * this->simulations_)]);
        
        this->node_p_ = static_cast<int*>(store);
    }

    size_t total_data_size() const
    {
        return sizeof(sim_metadata) * this->simulations_
             + sizeof(double)       * this->simulations_ * this->variables_count_
             + sizeof(int)          * this->simulations_ * this->models_count_;
    }

    result_pointers load_results() const
    {
        if(!this->is_cuda_) return result_pointers{
            false,
            nullptr,
            this->metadata_p_,
            this->node_p_,
            this->variable_p_
        };

        const size_t size = this->total_data_size();
        const void* source = this->metadata_p_; //this is the source of the array. nodes and variables are just offsets from here
        void* store = malloc(size);
        cudaMemcpy(store, source , size, cudaMemcpyDeviceToHost);

        sim_metadata* meta = static_cast<sim_metadata*>(store);
        store = static_cast<void*>(&this->metadata_p_[this->simulations_]);
        
        double* vars = static_cast<double*>(store);
        store = static_cast<void*>(&this->variable_p_[(this->variables_count_ * this->simulations_)]);
        
        int* nodes = static_cast<int*>(store);
        return result_pointers{
            true,
            store,
            meta,
            nodes,
            vars
        };
    }

    //void write_output is handled by state
    CPU GPU void write_output(const state* sim) const
    {
        const int sim_id = static_cast<int>(sim->simulation_id);

        this->metadata_p_[sim_id].steps = sim->steps; 
        this->metadata_p_[sim_id].global_time = sim->global_time;

        for (int i = 0; i < sim->variables.size; ++i)
        {
            if(!sim->variables.store[i].should_track) continue;
            this->variable_p_[sim_id*this->variables_count_ + i] = sim->variables.store[i].value;
        }

        for (int i = 0; i < sim->models.size; ++i)
        {
            this->node_p_[sim_id*this->models_count_ + i] =
                sim->models.store[i]->id
            * (!sim->models.store[i]->is_goal * (-1));
                // sim->models.store[i]->is_goal
                // ? sim->models.store[i]->id
                // : -sim->models.store[i]->id;
        }
    } 
};
