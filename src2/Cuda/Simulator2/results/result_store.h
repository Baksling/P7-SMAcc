#pragma once
#include <cstdlib>
#include "../common/macro.h"
#include "../allocations/memory_allocator.h"
#include "../Domain.h"


struct sim_metadata
{
    unsigned int steps;
    double global_time;
};

struct result_pointers
{
private:
    const bool owns_pointers_;
    void* source_p_;
public:
    explicit result_pointers(const bool owns_pointers, void* free_p,
        sim_metadata* results, int* nodes, double* variables);

    sim_metadata* meta_results = nullptr;
    int* nodes = nullptr;
    double* variables = nullptr;

    void free_internals() const;
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

    size_t total_data_size() const;
public:
    explicit result_store(const unsigned total_sim,
                          const unsigned variables,
                          const unsigned network_size,
                          memory_allocator* helper);

    result_pointers load_results() const;

    //This MUST be in here, in order not to cause RDC to be required.
    CPU GPU void write_output(const state* sim) const
    {
        const int sim_id = static_cast<int>(sim->simulation_id);

        this->metadata_p_[sim_id].steps = sim->steps; 
        this->metadata_p_[sim_id].global_time = sim->global_time;
        
        for (int i = 0, j = 0; i < sim->variables.size; ++i)
        {
            if(!sim->variables.store[i].should_track) continue;
            this->variable_p_[sim_id*this->variables_count_ + j++] = sim->variables.store[i].value;
        }

        for (int i = 0; i < sim->models.size; ++i)
        {
            // sim->models.store[i]->is_goal
            // ? sim->models.store[i]->id
            // : -sim->models.store[i]->id;

            // this->node_p_[sim_id*sim->models.size + i] = sim->models.store[i]->is_goal ? sim->models.store[i]->id : -sim->models.store[i]->id;
            this->node_p_[sim_id*sim->models.size + i] = sim->models.store[i]->id * ((sim->models.store[i]->is_goal*2) + (-1));
        }
    } 
};
