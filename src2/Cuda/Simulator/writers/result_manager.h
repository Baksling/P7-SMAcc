#pragma once

#include "../../common/macro.h"
#include "../../common/lend_array.h"
#include "simulation_result.h"
#include "../../Domain/stochastic_model_t.h"
#include "trace.h"
#include "../../Domain/simulator_state.h"
#include <string>

struct sim_pointers
{
private:
    const bool owns_pointers_;
public:
    explicit sim_pointers(const bool owns_pointers, simulation_result* results, int* nodes, double* variables);

    simulation_result* meta_results = nullptr;
    int* nodes = nullptr;
    double* variables = nullptr;

    void free_internals() const;
};

class result_manager
{
private:
    bool is_cuda_;
    
    unsigned int simulations_;
    unsigned int models_count_;
    unsigned int variables_count_;

    trace_interval interval_{};
    unsigned trace_data_size_;
    trace_vector* trace_data_;
    unsigned* trace_stack_p_;
    
    simulation_result* results_p_ = nullptr;
    double* variable_p_ = nullptr;
    int* node_p_ = nullptr;
        
    
    void load_results(
        simulation_result** out_r,
        int** out_n,
        double** out_v) const;

    CPU GPU void write_trace_to_stack(const unsigned sim_id, const trace_vector& data) const;

    explicit result_manager(
        const stochastic_model_t* model,
        const simulation_strategy* strategy,
        allocation_helper* allocator);
public:
    const bool tracking_trace;
    
    //SIMULATION methods
    CPU GPU void write_step_trace(const model_state* node, simulator_state* state) const;
    CPU GPU void write_node_trace(const model_state* node, const simulator_state* state) const;
    CPU GPU void write_result(const simulator_state* state) const;
    CPU GPU simulation_result* get_sim_results(const unsigned int sim_id) const;
    CPU GPU lend_array<int> get_nodes(const unsigned int sim_id) const;
    CPU GPU lend_array<double> get_variables(const unsigned int sim_id) const;

    //HOST methods
    sim_pointers analyse(std::unordered_map<int, node_result>* node_results, const array_t<variable_result>* var_results) const;
    void print_memory_usage(const allocation_helper* helper, size_t pre_allocate_size) const;
    void clear() const;
    trace_pointers load_trace() const;

    
    //FACTORY methods
    static result_manager* init(
        const stochastic_model_t* model,
        const simulation_strategy* strategy,
        allocation_helper* helper);
    
    static void init_unified(
        result_manager** host_handle,
        result_manager** device_handle,
        const stochastic_model_t* model,
        const simulation_strategy* strategy,
        allocation_helper* helper);

    static trace_interval parse_interval(const std::string& str);
};


