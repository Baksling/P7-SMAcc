#include "simulation_result.h"
#include "thread_pool.h"
#include "simulation_strategy.h"

void simulation_result_container::load_results(simulation_result** out_r, int** out_n, double** out_v) const
{
    if(!this->is_cuda_results_)
    {
        *out_r = this->results_p_;
        *out_n = this->node_p_;
        *out_v = this->variable_p_;
        return;
    }

    simulation_result* local_r = static_cast<simulation_result*>(malloc(sizeof(simulation_result)*this->results_count_));
    int* local_n               = static_cast<int*>(malloc(sizeof(int)*this->results_count_ * this->models_count_));
    double* local_v            = static_cast<double*>(malloc(sizeof(double)*this->results_count_ * this->variables_count_));

    cudaMemcpy(local_r, this->results_p_ , sizeof(simulation_result) * this->results_count_, cudaMemcpyDeviceToHost);
    cudaMemcpy(local_n, this->node_p_    , sizeof(int) * this->results_count_ * this->models_count_, cudaMemcpyDeviceToHost);
    cudaMemcpy(local_v, this->variable_p_, sizeof(double) * this->results_count_ * this->variables_count_, cudaMemcpyDeviceToHost);

    *out_r = local_r;
    *out_n = local_n;
    *out_v = local_v;
}

template<typename T>
CPU GPU void allocate(T** ptr, unsigned long long int size, const bool cuda_allocate)
{
    if(cuda_allocate)
    {
        cudaMalloc(ptr, size);
    }
    else
    {
        *ptr = static_cast<T*>(malloc(size));
    }
}

simulation_result_container::simulation_result_container(
    const unsigned size,
    const unsigned models,
    const unsigned variables,
    const bool cuda_allocate)
{
    this->is_cuda_results_ = cuda_allocate;
    this->results_count_ = size;
    this->models_count_ = models;
    this->variables_count_ = variables;

    allocate(&this->results_p_, sizeof(simulation_result)*size, this->is_cuda_results_);
    allocate(&this->variable_p_, sizeof(double)*size*variables, this->is_cuda_results_);
    allocate(&this->node_p_, sizeof(int)*size*models, this->is_cuda_results_);
}

void simulation_result_container::free_internals() const
{
    if(this->is_cuda_results_)
    {
        cudaFree(this->results_p_);
        cudaFree(this->variable_p_);
        cudaFree(this->node_p_);
    }
    else
    {
        free(this->results_p_);
        free(this->variable_p_);
        free(this->node_p_);
    }
}


sim_pointers simulation_result_container::analyse(
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
    
    for (unsigned  i = 0; i < this->results_count_; ++i)
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

    return sim_pointers{ this->is_cuda_results_, local_results, local_nodes, local_variables };
}

simulation_result_container* simulation_result_container::cuda_allocate(const allocation_helper* helper) const
{
    simulation_result_container* p = nullptr; 
    cudaMalloc(&p, sizeof(simulation_result_container));
    helper->free_list->push_back(p);
    cudaMemcpy(p, this, sizeof(simulation_result_container), cudaMemcpyHostToDevice);
    return p;
}

CPU GPU simulation_result* simulation_result_container::get_sim_results(const unsigned sim_id) const
{
    if(sim_id >= this->results_count_) printf("Inaccessible result location");
    return &this->results_p_[sim_id];
}

CPU GPU lend_array<int> simulation_result_container::get_nodes(const unsigned sim_id) const
{
    if(sim_id >= this->results_count_) printf("Inaccessible node result location");
    const unsigned long long i = static_cast<unsigned long long>(this->models_count_) * sim_id;
    return lend_array<int>(&this->node_p_[i], static_cast<int>(this->models_count_));
}

CPU GPU lend_array<double> simulation_result_container::get_variables(const unsigned sim_id) const
{
    if(sim_id >= this->results_count_) printf("Inaccessible variable result location");
    const unsigned long long i = static_cast<unsigned long long>(this->variables_count_) * sim_id;
    return lend_array<double>(&this->variable_p_[i], static_cast<int>(this->variables_count_));
}
