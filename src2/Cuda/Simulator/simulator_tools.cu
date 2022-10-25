#include "simulator_tools.h"

#include "../Domain/edge_t.h"
#include "../Domain/simulator_state.h"
#include "../common/allocation_helper.h"
#include "../common/array_t.h"
#include "../common/lend_array.h"


CPU GPU bool simulator_tools::bit_is_set(const unsigned long long* n, const unsigned int i)
{
    return (*n) & (1UL << i);
}


CPU GPU void simulator_tools::set_bit(unsigned long long* n, const unsigned int i)
{
    (*n) |=  (1UL << (i));
}

CPU GPU void simulator_tools::unset_bit(unsigned long long* n, const unsigned int i)
{
    (*n) &= ~(1UL << (i));
}

array_t<variable_result> simulator_tools::allocate_variable_results(const unsigned variable_count)
{
    const array_t<variable_result> variables = array_t<variable_result>( static_cast<int>(variable_count));
    for (unsigned int i = 0; i < variable_count; ++i)
    {
        variables.arr()[i] = variable_result{i, 0, 0};
    }
    return variables;
}


simulation_result* simulator_tools::allocate_results(const simulation_strategy* strategy,
                                                    const unsigned int variable_count,
                                                    const unsigned int models_count, 
                                                    std::list<void*>* free_list,
                                                    const bool cuda_allocate)
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

        int* node_results = nullptr;
        if(cuda_allocate) cudaMalloc(&node_results, sizeof(int)*models_count);
        else node_results = static_cast<int*>(malloc(sizeof(int)*models_count));

        free_list->push_back(node_results);
        free_list->push_back(variable_results);
        local_results[i] = simulation_result{
            0,
            0.0,
            node_results,
            variable_results };
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

void simulator_tools::read_results(
    const simulation_result* simulation_results,
    const unsigned long total_simulations,
    const unsigned int model_count,
    std::map<int, node_result>* results,
    const lend_array<variable_result>* avg_max_variable_value,
    const bool cuda_results)
{
    const unsigned long long variable_size = sizeof(double) * avg_max_variable_value->size();
    const unsigned long long node_size = sizeof(int) * model_count;

    double* local_variable_results = static_cast<double*>(malloc(variable_size));
    int* local_node_results = static_cast<int*>(malloc(node_size));
    
    results->insert( std::pair<int, node_result>(HIT_MAX_STEPS, node_result{ 0, 0.0 }) );
    
    for (unsigned long i = 0; i < total_simulations; ++i)
    {
        const simulation_result result = simulation_results[i];

        if(cuda_results) //copy from cuda
            cudaMemcpy(local_node_results, result.end_node_id_arr, node_size, cudaMemcpyDeviceToHost);
        else //set variable from local
            // local_node_results = result.end_node_id_arr;
            cudaMemcpy(local_node_results, result.end_node_id_arr, node_size, cudaMemcpyHostToHost);
        
        for (unsigned j = 0; j < model_count; ++j)
        {
            if(results->count(local_node_results[j]) == 1)
            {
                results->at(local_node_results[j]).update_count(result.steps);
            }
            else
            {
                int id = local_node_results[j];
                node_result r = {1, static_cast<double>(result.steps)};
                results->insert( std::pair<int, node_result>(id, r) );
            }
        }
        
        if(cuda_results) //copy from cuda
            cudaMemcpy(local_variable_results, result.variables_max_value_arr, variable_size, cudaMemcpyDeviceToHost);
        else //set variable from local
            memcpy(local_variable_results, result.variables_max_value_arr, variable_size);


        for (int  k = 0; k < avg_max_variable_value->size(); ++k)
        {
            avg_max_variable_value->at(k)->update_count(local_variable_results[k]);
        }  
    }
    free(local_variable_results);
    free(local_node_results);
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

void simulator_tools::print_results(const std::map<int, node_result>* result_map,
                                   const lend_array<variable_result>* variable_results, const unsigned long total_simulations)
{
    //Variables
    std::cout << "\naverage maximum value of each variable: \n";
    for (int i = 0; i < variable_results->size(); ++i)
    {
        const variable_result* result = variable_results->at(i);
        std::cout << "Variable " << result->variable_id << " = " << result->avg_max_value << "\n";
    }

    //Nodes
    std::cout << "\n\ngoal nodes: \n";
    for (const std::pair<const int, node_result>& it : (*result_map))
    {
        if(it.first == HIT_MAX_STEPS) continue;
        const float percentage = calc_percentage(it.second.reach_count, total_simulations);
        print_node(it.first, it.second.reach_count, percentage,it.second.avg_steps);
    }

    //No goal state reached
    const node_result no_value = result_map->at(HIT_MAX_STEPS);
    const float percentage = calc_percentage(no_value.reach_count, total_simulations);

    std::cout << "No goal node was reached " << no_value.reach_count << " times. ("
            << percentage << ")%. avg step count: " << no_value.avg_steps << ".\n";

    //Total sim
    std::cout << "\n";
    std::cout << "Nr of simulations: " << total_simulations << "\n";
    std::cout << "\n";

}


CPU GPU edge_t* find_valid_edge_heap(
    simulator_state* state,
    const lend_array<edge_t*>* edges,
    curandState* r_state)
{
    // return nullptr;
    edge_t** valid_edges = static_cast<edge_t**>(malloc(sizeof(edge_t*) * edges->size()));  // NOLINT(bugprone-sizeof-expression)
    if(valid_edges == nullptr) printf("COULD NOT ALLOCATE HEAP MEMORY\n");
    int valid_count = 0;
    
    for (int i = 0; i < edges->size(); ++i)
    {
        valid_edges[i] = nullptr; //clean malloc
        edge_t* edge = edges->get(i);
        if(edge->evaluate_constraints(state))
            valid_edges[valid_count++] = edge;
    }
    
    if(valid_count == 0)
    {
        free(valid_edges);
        return nullptr;
    }
    if(valid_count == 1)
    {
        edge_t* result = valid_edges[0];
        free(valid_edges);
        return result;
    }

    //summed weight
    double weight_sum = 0.0;
    for(int i = 0; i < valid_count; i++)
    {
        weight_sum += valid_edges[i]->get_weight(state);
    }

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const double r_val = (1.0 - curand_uniform_double(r_state))*weight_sum;
    double r_acc = 0.0; 

    //pick the weighted random value.
    for (int i = 0; i < valid_count; ++i)
    {
        edge_t* temp = valid_edges[i];
        r_acc += temp->get_weight(state);
        if(r_val < r_acc)
        {
            free(valid_edges);
            return temp;
        }
    }

    //This should be handled in for loop.
    //This is for safety :)
    edge_t* edge = valid_edges[valid_count - 1];
    free(valid_edges);
    return edge;
}

CPU GPU edge_t* find_valid_edge_fast(
    simulator_state* state,
    const lend_array<edge_t*>* edges,
    curandState* r_state)
{
    unsigned long long valid_edges_bitarray = 0UL;
    unsigned int valid_count = 0;
    edge_t* valid_edge = nullptr;
    
    for (int i = 0; i < edges->size(); ++i)
    {
        edge_t* edge = edges->get(i);
        if(edge->evaluate_constraints(state))
        {
            simulator_tools::set_bit(&valid_edges_bitarray, i);
            valid_edge = edge;
            valid_count++;
        }
    }

    if(valid_count == 0) return nullptr;
    if(valid_count == 1 && valid_edge != nullptr) return valid_edge;
    
    //summed weight
    double weight_sum = 0.0;
    for(int i = 0; i  < edges->size(); i++)
    {
        if(simulator_tools::bit_is_set(&valid_edges_bitarray, i))
            weight_sum += edges->get(i)->get_weight(state);
    }

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const double r_val = (1.0 - curand_uniform_double(r_state)) * weight_sum;
    double r_acc = 0.0; 

    //pick the weighted random value.
    valid_edge = nullptr; //reset valid edge !IMPORTANT
    for (int i = 0; i < edges->size(); ++i)
    {
        if(!simulator_tools::bit_is_set(&valid_edges_bitarray, i)) continue;

        valid_edge = edges->get(i);
        r_acc += valid_edge->get_weight(state);
        if(r_val < r_acc)
        {
            return valid_edge;
        }
    }
    return valid_edge;
}


CPU GPU edge_t* simulator_tools::choose_next_edge(simulator_state* state, const lend_array<edge_t*>* edges, curandState* r_state)
{
    //if no possible edges, return null pointer
    if(edges->size() == 0) return nullptr;
    if(edges->size() == 1)
    {
        edge_t* edge = edges->get(0);
        return edge->evaluate_constraints(state)
                ? edge
                : nullptr;
    }

    if(static_cast<unsigned long long>(edges->size()) < sizeof(unsigned long long)*8)
        return find_valid_edge_fast(state, edges, r_state);
    else
        return find_valid_edge_heap(state, edges, r_state);
}