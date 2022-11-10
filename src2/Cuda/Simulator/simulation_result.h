#pragma once
#include <unordered_map>

#include "../common/allocation_helper.h"
#include "../common/lend_array.h"

struct node_result
{
    unsigned int reach_count;
    double avg_steps;

    void update_count(const unsigned int avg_step)
    {
        avg_steps = ((avg_steps * reach_count) + static_cast<double>(avg_step)) / (reach_count+1);
        reach_count++;
    }
};

struct variable_result
{
    unsigned int variable_id;
    double avg_max_value;
    unsigned long values_counted;

    void update_count(const double max_value)
    {
        avg_max_value = ((avg_max_value * values_counted) + max_value) / (values_counted+1);
        values_counted++;
    }
};

struct simulation_result
{
    unsigned int steps;
    double total_time_progress;
};

struct sim_pointers
{
private:
    bool owns_pointers_;
public:
    explicit sim_pointers(const bool owns_pointers, simulation_result* results, int* nodes, double* variables)
    {
        this->owns_pointers_ = owns_pointers;
        this->meta_results = results;
        this->nodes = nodes;
        this->variables = variables;
    }
    
    simulation_result* meta_results = nullptr;
    int* nodes = nullptr;
    double* variables = nullptr;

    void free_internals() const
    {
        if(!owns_pointers_) return;
        free(this->meta_results);
        free(this->nodes);
        free(this->variables);
    }
};


class simulation_result_container
{
private:
    bool is_cuda_results_;
    
    unsigned int results_count_;
    unsigned int models_count_;
    unsigned int variables_count_;
    
    simulation_result* results_p_{};
    double* variable_p_{};
    int* node_p_{};

    void load_results(
        simulation_result** out_r,
        int** out_n,
        double** out_v) const;
public:
    explicit simulation_result_container(
        const unsigned size,
        const unsigned models,
        const unsigned variables,
        allocation_helper* cuda_allocate);

    sim_pointers analyse(std::unordered_map<int, node_result>* node_results, const array_t<variable_result>* var_results) const;

    simulation_result_container* cuda_allocate(allocation_helper* helper) const; 
    
    CPU GPU simulation_result* get_sim_results(const unsigned int sim_id) const;
    CPU GPU lend_array<int> get_nodes(const unsigned int sim_id) const;
    CPU GPU lend_array<double> get_variables(const unsigned int sim_id) const;
};
