#include "simulator_tools.h"

#include "../common/allocation_helper.h"
#include "../common/array_t.h"
#include "../common/lend_array.h"


CPU GPU bool bit_handler::bit_is_set(const unsigned long long* n, const unsigned int i)
{
    return (*n) & (1UL << i);
}


CPU GPU void bit_handler::set_bit(unsigned long long* n, const unsigned int i)
{
    (*n) |=  (1UL << (i));
}

CPU GPU void bit_handler::unset_bit(unsigned long long* n, const unsigned int i)
{
    (*n) &= ~(1UL << (i));
}

array_t<variable_result> result_handler::allocate_variable_results(const unsigned variable_count)
{
    const array_t<variable_result> variables = array_t<variable_result>( static_cast<int>(variable_count));
    for (unsigned int i = 0; i < variable_count; ++i)
    {
        variables.arr()[i] = variable_result{i, 0, 0};
    }
    return variables;
}


simulation_result* result_handler::allocate_results(const simulation_strategy* strategy,
                                                    const unsigned int variable_count, const allocation_helper* helper, bool cuda_allocate)
{
    const unsigned long total = strategy->total_simulations();
    const unsigned long long mem_size = sizeof(simulation_result) * total;
    
    simulation_result* local_results = static_cast<simulation_result*>(malloc(mem_size));

    for (unsigned long i = 0; i < total; ++i)
    {
        //allocate variables
        double* variable_results = nullptr;
        if(cuda_allocate) cudaMalloc(&variable_results, sizeof(double)*variable_count);
        else variable_results = static_cast<double*>(malloc(sizeof(double)*variable_count));
        
        helper->free_list->push_back(variable_results);
        local_results[i] = simulation_result{ HIT_MAX_STEPS, 0, variable_results };
    }

    simulation_result* results = nullptr; 
    if(cuda_allocate)
    {
        cudaMalloc(&results, mem_size);
        cudaMemcpy(results, local_results, mem_size, cudaMemcpyHostToDevice);
        free(local_results);
    }
    else
    {
        results = local_results;
    }
    
    return results;
}

void result_handler::read_results(const simulation_result* simulation_results, const unsigned long total_simulations,
    std::map<int, node_result>* results, const lend_array<variable_result>* avg_max_variable_value)
{
    const unsigned long long variable_size = sizeof(double) * avg_max_variable_value->size();
    double* local_variable_results = static_cast<double*>(malloc(variable_size));

    results->insert( std::pair<int, node_result>(HIT_MAX_STEPS, node_result{ 0, 0.0 }) );
    
    for (unsigned long i = 0; i < total_simulations; ++i)
    {
        const simulation_result result = simulation_results[i];
        if(results->count(result.id) == 1)
        {
            results->at(result.id).update_count(result.steps);
        }
        else
        {
            node_result r = {1, static_cast<double>(result.steps)};
            results->insert( std::pair<int, node_result>(result.id, r) );
        }
        
        cudaMemcpy(local_variable_results, result.variables_max_value, variable_size, cudaMemcpyDeviceToHost);

        for (int  j = 0; j < avg_max_variable_value->size(); ++j)
        {
            avg_max_variable_value->at(j)->update_count(local_variable_results[j]);
        }
    }
    free(local_variable_results);
}

float calc_percentage(const unsigned long counter, const unsigned long divisor)
{
    return (static_cast<float>(counter)/static_cast<float>(divisor))*100;
}


void print_node(const int node_id, const unsigned int reached_count, const float reach_percentage, const double avg_steps)
{
    std::cout << "Node: " << node_id << " reached " << reached_count << " times. ("
            << reach_percentage << ")%. avg step count: " << avg_steps << ".\n";
}

void result_handler::print_results(std::map<int, node_result>* result_map,
    const lend_array<variable_result>* variable_results, const unsigned long result_size)
{
    //Variables
    std::cout << "\naverage maximum value of each variable: ";
    for (int i = 0; i < variable_results->size(); ++i)
    {
        const variable_result* result = variable_results->at(i);
        std::cout << "Variable: " << result->variable_id << " = " << result->avg_max_value;
    }

    //Nodes
    std::cout << "\n\ngoal nodes: \n";
    for (const std::pair<const int, node_result>& it : (*result_map))
    {
        if(it.first == HIT_MAX_STEPS) continue;
        const float percentage = calc_percentage(it.second.reach_count, result_size);
        print_node(it.first, it.second.reach_count, percentage,it.second.avg_steps);
    }

    //No goal state reached
    const node_result no_value = result_map->at(HIT_MAX_STEPS);
    const float percentage = calc_percentage(no_value.reach_count, result_size);

    std::cout << "No goal node was reached " << no_value.reach_count << " times. ("
            << percentage << ")%. avg step count: " << no_value.avg_steps << ".\n";
    print_node(HIT_MAX_STEPS, no_value.reach_count, percentage, no_value.avg_steps);

    //Total sim
    std::cout << "\n";
    std::cout << "Nr of simulations: " << result_size << "\n";
    std::cout << "\n";

}