﻿#include "result_writer.h"

#include <cuda.h>
#include <iostream>
#include <fstream>

using namespace std::chrono;

void result_writer::analyse_results(const simulation_result* simulation_results, const unsigned long total_simulations,
    std::map<int, node_result>* results, const array_t<variable_result>* avg_max_variable_value, bool from_cuda) const
{
    const unsigned long long node_size = sizeof(int) * this->model_count_;

    const cudaMemcpyKind kind = from_cuda ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
    
    double* local_variable_results = static_cast<double*>(malloc(sizeof(double) * avg_max_variable_value->size()));
    int* local_node_results = static_cast<int*>(malloc(node_size));
    
    results->insert( std::pair<int, node_result>(HIT_MAX_STEPS, node_result{ 0, 0.0 }));
    
    for (unsigned long i = 0; i < total_simulations; ++i)
    {
        const simulation_result local_result = simulation_results[i];
        
        cudaMemcpy(local_node_results, local_result.end_node_id_arr, node_size, kind);
        
        for (unsigned int j = 0; j < this->model_count_ ; ++j)
        {
            if(results->count(local_node_results[j]) == 1)
            {
                results->at(local_node_results[j]).update_count(local_result.steps);
            }
            else
            {
                node_result r = {1, static_cast<double>(local_result.steps)};
                results->insert( std::pair<int, node_result>(local_node_results[j], r) );
            }
        }
        
        cudaMemcpy(local_variable_results, local_result.variables_max_value_arr, sizeof(double) * avg_max_variable_value->size(), kind);

        for (int  j = 0; j < avg_max_variable_value->size(); ++j)
        {
            avg_max_variable_value->at(j)->update_count(local_variable_results[j]);
        }
    }
    free(local_variable_results);
    free(local_node_results);
}

float result_writer::calc_percentage(const unsigned long counter, const unsigned long divisor)
{
    return (static_cast<float>(counter)/static_cast<float>(divisor))*100;
}

std::string result_writer::print_node(const int node_id, const unsigned reached_count, const float reach_percentage,
                                      const double avg_steps)
{
    return "Node: " + std::to_string(node_id) + " reached " + std::to_string(reached_count) + " times. ("
            + std::to_string(reach_percentage) + ")%. avg step count: " + std::to_string(avg_steps) + ".\n"; 
}

void result_writer::write_to_file(const simulation_result* sim_result, std::map<int, node_result>* results,
    unsigned long total_simulations, array_t<variable_result> var_result, unsigned variable_count) const
{
    std::ofstream file_node, file_variable, summary;
    summary.open(this->file_path_ + ".summary.txt");
    
    summary << "\naverage maximum value of each variable: \n";
    
    for (int j = 0; j < var_result.size(); ++j)
    {
        const variable_result* result = var_result.at(j);
        summary << "variable " << result->variable_id << " = " << result->avg_max_value << "\n";
    }
    
    summary << "\n\ngoal: \n";

    for (const std::pair<const int, node_result>& it : *results)
    {
        if(it.first == HIT_MAX_STEPS) continue;
        const float percentage = calc_percentage(it.second.reach_count, total_simulations);
        summary << print_node(it.first, it.second.reach_count, percentage, it.second.avg_steps);
    }

    const node_result no_value = results->at(HIT_MAX_STEPS);
    const float percentage = calc_percentage(no_value.reach_count, total_simulations);

    summary << "No goal node was reached " << no_value.reach_count << " times. ("
         << percentage << ")%. avg step count: " << no_value.avg_steps << ".\n";

    summary << "\nNr of simulations: " << total_simulations << "\n\n";

    summary.flush();
    summary.close();

    file_node.open(this->file_path_ + "_node_data.tsv", std::ios::app);
    file_variable.open(this->file_path_ + "_variable_data.tsv", std::ios::app);

    file_node << "Simulation" << "\t" << "Model" << "\t" << "Node" << "\n";
    file_variable << "Simulation" << "\t" << "Variable" << "\t" << "Value" << "\n";
    
    for (unsigned i = 0; i < total_simulations; ++i)
    {
        for (unsigned j = 0; j < this->model_count_; ++j)
        {
            file_node << i << "\t" << j << "\t" << sim_result->end_node_id_arr[j] << "\n";
        }

        for (unsigned k = 0; k < variable_count; ++k)
        {
            file_variable << i << "\t" << k << "\t" << sim_result->variables_max_value_arr[k]<< "\n";
        }
        
        file_node.flush();
        file_variable.flush();

        file_node.close();
        file_variable.close();
    }
}

void result_writer::write_to_console(std::map<int, node_result>* results, unsigned long total_simulations,
    array_t<variable_result> var_result) const
{
    
    printf("\naverage maximum value of each variable: \n");
    
    for (int j = 0; j < var_result.size(); ++j)
    {
        const variable_result* result = var_result.at(j);
        std::cout << "variable " << result->variable_id << " = " << result->avg_max_value << "\n";
    }
    
    std::cout << "\n\ngoal: \n";

    for (const std::pair<const int, node_result>& it : *results)
    {
        if(it.first == HIT_MAX_STEPS) continue;
        const float percentage = calc_percentage(it.second.reach_count, total_simulations);
        std::cout << print_node(it.first, it.second.reach_count, percentage, it.second.avg_steps);
    }

    const node_result no_value = results->at(HIT_MAX_STEPS);
    const float percentage = calc_percentage(no_value.reach_count, total_simulations);

    std::cout << "No goal node was reached " << no_value.reach_count << " times. ("
         << percentage << ")%. avg step count: " << no_value.avg_steps << ".\n";

    std::cout << "\nNr of simulations: " << total_simulations << "\n\n";
}

result_writer::result_writer(const std::string* path,const simulation_strategy strategy, unsigned model_count,
                             const bool write_to_console, const bool write_to_file)
{
    this->file_path_ = *path;
    this->strategy_ = strategy;
    this->write_to_console_ = write_to_console;
    this->write_to_file_ = write_to_file;
    this->model_count_ = model_count;
}

void result_writer::write_results(const simulation_result* sim_result, const unsigned result_size,
    const unsigned variable_count, steady_clock::duration sim_duration, bool from_cuda) const
{       
    std::map<int, node_result> result_map;
    const array_t<variable_result> var_result = array_t<variable_result>(static_cast<int>(variable_count));

    for (unsigned int k = 0; k < static_cast<unsigned int>(var_result.size()); ++k)
    {
        var_result.arr()[k] = variable_result{k,0,0};
    }
    
    analyse_results(sim_result, strategy_.total_simulations(), &result_map, &var_result, from_cuda);
    
    if (this->write_to_file_) write_to_file(sim_result, &result_map, strategy_.total_simulations(), var_result, variable_count);
    if (this->write_to_console_) write_to_console(&result_map, strategy_.total_simulations(), var_result);
}