#include "result_manager.h"
#include "../simulation_strategy.h"
#include <string>
#include <iostream>

sim_pointers::sim_pointers(const bool owns_pointers, simulation_result* results, int* nodes, double* variables):
    owns_pointers_(owns_pointers)
{
    this->meta_results = results;
    this->nodes = nodes;
    this->variables = variables;
}

void sim_pointers::free_internals() const
{
    if(!owns_pointers_) return;
    free(this->meta_results);
    free(this->nodes);
    free(this->variables);
}

trace_pointers result_manager::load_trace() const
{
    if(!this->is_cuda_)
    {
        return trace_pointers{
            false,
            this->simulations_,
            this->trace_data_size_,
            this->trace_stack_p_,
            this->trace_data_
        };
    }
    const size_t stack_count_size = sizeof(unsigned)*this->simulations_;
    const size_t data_size = sizeof(trace_vector)*this->trace_data_size_*this->simulations_;
        
    unsigned* stack_count_p = static_cast<unsigned*>(malloc(stack_count_size));
    trace_vector* data_p = static_cast<trace_vector*>(malloc(data_size));

    cudaMemcpy(stack_count_p, this->trace_stack_p_, stack_count_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(data_p, this->trace_data_, data_size, cudaMemcpyDeviceToHost);

    return trace_pointers{
        true,
        this->simulations_,
        this->trace_data_size_,
        stack_count_p,
        data_p
    };
}

void result_manager::load_results(simulation_result** out_r, int** out_n, double** out_v) const
{
    if(!this->is_cuda_)
    {
        *out_r = this->results_p_;
        *out_n = this->node_p_;
        *out_v = this->variable_p_;
        return;
    }

    simulation_result* local_r = static_cast<simulation_result*>(malloc(sizeof(simulation_result)*this->simulations_));
    int* local_n               = static_cast<int*>(malloc(sizeof(int)*this->simulations_ * this->models_count_));
    double* local_v            = static_cast<double*>(malloc(sizeof(double)*this->simulations_ * this->variables_count_));

    cudaMemcpy(local_r, this->results_p_ , sizeof(simulation_result) * this->simulations_, cudaMemcpyDeviceToHost);
    cudaMemcpy(local_n, this->node_p_    , sizeof(int) * this->simulations_ * this->models_count_, cudaMemcpyDeviceToHost);
    cudaMemcpy(local_v, this->variable_p_, sizeof(double) * this->simulations_ * this->variables_count_, cudaMemcpyDeviceToHost);
    
    *out_r = local_r;
    *out_n = local_n;
    *out_v = local_v;
}

CPU GPU void result_manager::write_trace_to_stack(const unsigned sim_id, const trace_vector& data) const
{
    unsigned* stack_p = &this->trace_stack_p_[sim_id];
    this->trace_data_[this->trace_data_size_ * sim_id + (*stack_p)] = data;
    (*stack_p) += 1;
}

result_manager::result_manager(
    const stochastic_model_t* model,
    const simulation_strategy* strategy,
    allocation_helper* allocator) : tracking_trace(strategy->trace_settings.mode != trace_interval::disabled)
{
    this->is_cuda_ = allocator->use_cuda;
    this->interval_ = strategy->trace_settings;
    this->simulations_ = strategy->total_simulations();
    this->models_count_ = model->get_models_count();
    this->variables_count_ = model->get_variable_count();
    this->trace_data_size_ = strategy->max_sim_steps * model->get_models_count() * model->get_variable_count();
    
    allocator->allocate(&this->results_p_, sizeof(simulation_result) * this->simulations_);
    allocator->allocate(&this->variable_p_, sizeof(double) * this->simulations_ * this->variables_count_);
    allocator->allocate(&this->node_p_, sizeof(int) * this->simulations_ * this->models_count_);
    
    if(this->interval_.mode != trace_interval::disabled)
    {
        allocator->allocate(&this->trace_data_, sizeof(trace_vector)* this->simulations_ * this->trace_data_size_);
        allocator->allocate(&this->trace_stack_p_, sizeof(unsigned)*this->simulations_);
        this->clear();
    }
    else
    {
        this->trace_data_ = nullptr;
        this->trace_stack_p_ = nullptr;
    }
}

CPU GPU void result_manager::write_step_trace(const model_state* node, simulator_state* state)
{
    if(this->interval_.mode == trace_interval::disabled) return; //trace tracking disabled
    if(this->interval_.mode == trace_interval::time_interval && state->trace_time_ < this->interval_.value ) return;
    if(this->interval_.mode == trace_interval::step_interval
        && (state->steps_ % static_cast<unsigned long long>(this->interval_.value) != 0)) return;

    this->write_node_trace(node, state);

    state->trace_time_ += this->interval_.value;
    
    for (int i = 0; i < state->variables_.size(); ++i)
    {
        const trace_vector data = trace_vector{
            state->steps_,
            static_cast<unsigned>(i),
            state->variables_.at(i)->get_time(),
            false
        };
            
        this->write_trace_to_stack(state->sim_id_, data);
    }
}

CPU GPU void result_manager::write_node_trace(const model_state* node, const simulator_state* state) const
{
    if(this->interval_.mode == trace_interval::disabled) return; //trace tracking disabled
    if(this->interval_.mode == trace_interval::time_interval && state->global_time_ < this->interval_.value ) return;
    if(this->interval_.mode == trace_interval::step_interval
        && (state->steps_ % static_cast<size_t>(this->interval_.value) != 0)) return;
    
    const trace_vector data = trace_vector{
        state->steps_,
        static_cast<unsigned>(node->current_node->get_id()),
        state->global_time_,
        true
    };

    this->write_trace_to_stack(state->sim_id_, data);
}

CPU GPU void result_manager::write_result(const simulator_state* state) const
{
    simulation_result* output = this->get_sim_results(state->sim_id_);

    printf("Hababa: %u\n", state->sim_id_);
    
    output->total_time_progress = 0; //state->global_time_;
    output->steps = 1; // state->steps_;

    const lend_array<int> node_results = this->get_nodes(state->sim_id_);
    const lend_array<double> var_results = this->get_variables(state->sim_id_);

    if(node_results.size() != state->models_.size() || var_results.size() != state->variables_.size())
    {
        printf("Expected number of models or variables does not match actual amount of models/variables!\n");
        return;
    }
    for (int i = 0; i < node_results.size(); ++i)
    {
        int* p = node_results.at(i);
        (*p) = state->models_.at(i)->reached_goal
            ? state->models_.at(i)->current_node->get_id()
            : HIT_MAX_STEPS;
    }

    for (int i = 0; i < var_results.size(); ++i)
    {
        double* p = var_results.at(i);
        // continue;
        *p = state->variables_.at(i)->get_max_value();
    }
}

sim_pointers result_manager::analyse(
    std::unordered_map<int, node_result>* node_results,
    const array_t<variable_result>* var_results
    ) const
{
    simulation_result* local_results = nullptr;
    int* local_nodes = nullptr;
    double* local_variables = nullptr;
    this->load_results(&local_results, &local_nodes, &local_variables);

    if (node_results->count(HIT_MAX_STEPS) == 0)
    {
        node_results->insert(std::pair<int, node_result>(HIT_MAX_STEPS, node_result{ 0, 0 }));
    }
    
    for (unsigned  i = 0; i < this->simulations_; ++i)
    {
        const simulation_result sim_r = local_results[i];

        for (unsigned j = 0; j < this->models_count_; ++j)
        {
            
            const int p = local_nodes[i * this->models_count_ + j];
            if(node_results->count(p) == 1)
            {
                node_results->at(p).update_count(sim_r.steps);
            }
            else
            {
                node_result nr = {1, static_cast<double>(sim_r.steps)};
                node_results->insert( std::pair<int, node_result>(p, nr) );
            }
        }

        for (unsigned k = 0; k < this->variables_count_; ++k)
        {
            const double val = local_variables[i * this->variables_count_ + k];
            var_results->at(static_cast<int>(k))->update_count(val);
        }
    }

    return sim_pointers{ this->is_cuda_, local_results, local_nodes, local_variables };
}


CPU GPU simulation_result* result_manager::get_sim_results(const unsigned sim_id) const
{
    if(sim_id >= this->simulations_) printf("Inaccessible result location");
    return &this->results_p_[sim_id];
}

CPU GPU lend_array<int> result_manager::get_nodes(const unsigned sim_id) const
{
    if(sim_id >= this->simulations_) printf("Inaccessible node result location");
    const unsigned long long i = static_cast<unsigned long long>(this->models_count_) * sim_id;
    return lend_array<int>(&this->node_p_[i], static_cast<int>(this->models_count_));
}

CPU GPU lend_array<double> result_manager::get_variables(const unsigned sim_id) const
{
    if(sim_id >= this->simulations_) printf("Inaccessible variable result location");
    const unsigned long long i = static_cast<unsigned long long>(this->variables_count_) * sim_id;
    return lend_array<double>(&this->variable_p_[i], static_cast<int>(this->variables_count_));
}

void result_manager::print_memory_usage(const allocation_helper* helper, const size_t pre_allocate_size) const
{
    const size_t mem_usage = helper->allocated_size - pre_allocate_size;

    if(helper->use_cuda)
    {
        size_t total_mem, free_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("output cache utilizes %llu / %llu bytes (%lf%%) (trace: %s)\n",
            static_cast<unsigned long long>(mem_usage / 8),
            static_cast<unsigned long long>(total_mem / 8),
            (static_cast<double>(mem_usage) / static_cast<double>(total_mem))*100,
            this->interval_.mode != trace_interval::disabled ? "Yes" : "No");
    }
    else
    {
        printf("output cache utilizes %llu bytes (trace: %s)\n",
            static_cast<unsigned long long>(mem_usage / 8),
            this->interval_.mode != trace_interval::disabled ? "Yes" : "No"
            );
    }
}

void result_manager::clear() const
{
    if(this->interval_.mode == trace_interval::disabled || this->trace_stack_p_ == nullptr) return;
    if(this->is_cuda_)
        cudaMemset(this->trace_stack_p_, 0, sizeof(unsigned)*this->simulations_);
    else
        memset(this->trace_stack_p_, 0, sizeof(unsigned)*this->simulations_);
}

result_manager* result_manager::init(
    const stochastic_model_t* model,
    const simulation_strategy* strategy,
    allocation_helper* helper)
{
    const result_manager instance = result_manager(
        model,
        strategy,
        helper);

    result_manager* p = nullptr;
    const cudaMemcpyKind kind = helper->use_cuda ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    helper->allocate(&p, sizeof(result_manager));
    cudaMemcpy(p, &instance, sizeof(result_manager), kind);
    return p;
}

void result_manager::init_unified(result_manager** host_handle, result_manager** device_handle,
    const stochastic_model_t* model,
    const simulation_strategy* strategy,
    allocation_helper* helper)
{
    helper->allocate(device_handle, sizeof(result_manager));
    *host_handle = new result_manager(
        model,
        strategy,
        helper);

    const cudaMemcpyKind kind = helper->use_cuda ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    cudaMemcpy(*device_handle, *host_handle, sizeof(result_manager), kind);
}

trace_interval result_manager::parse_interval(const std::string& str)
{
    trace_interval interval = trace_interval{ trace_interval::step_interval, 1.0 };
    if(str.empty()) return interval;

    if(str.back() == 't') interval.mode = trace_interval::time_interval;
    else if(str.back() == 's') interval.mode = trace_interval::step_interval;
    else if(str.back() == 'd') interval.mode = trace_interval::disabled;

    try
    {
        interval.value = std::stod(str.substr(0, str.size()-1));
    }
    catch (std::invalid_argument&)
    {
        printf("Could nto parse parse interval as double.");
        throw;
    }
    return interval;
}
