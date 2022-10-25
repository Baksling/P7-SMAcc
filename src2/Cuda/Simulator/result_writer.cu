#include "result_writer.h"

#include <cuda.h>
#include <iostream>
#include <fstream>

using namespace std::chrono;

void result_writer::analyse_results(const simulation_result* simulation_results, const unsigned long total_simulations,
    std::map<int, node_result>* results, const array_t<variable_result>* avg_max_variable_value)
{
    double* local_variable_results = static_cast<double*>(malloc(sizeof(double) * avg_max_variable_value->size()));
    
    results->insert( std::pair<int, node_result>(HIT_MAX_STEPS, node_result{ 0, 0.0 }));
    
    for (unsigned long i = 0; i < total_simulations; ++i)
    {
        // const simulation_result local_result = simulation_results[i];
        // if(results->count(local_result.id) == 1)
        // {
        //     results->at(local_result.id).update_count(local_result.steps);
        // }
        // else
        // {
        //     node_result r = {1, static_cast<double>(local_result.steps)};
        //     results->insert( std::pair<int, node_result>(local_result.id, r) );
        // }
        //
        // cudaMemcpy(local_variable_results, local_result.variables_max_value, sizeof(double) * avg_max_variable_value->size(), cudaMemcpyDeviceToHost);

        for (int  j = 0; j < avg_max_variable_value->size(); ++j)
        {
            avg_max_variable_value->at(j)->update_count(local_variable_results[j]);
        }
    }
    free(local_variable_results);
}

float result_writer::calc_percentage(const unsigned long counter, const unsigned long divisor)
{
    return (static_cast<float>(counter)/static_cast<float>(divisor))*100;
}

void result_writer::print_node(const int node_id, const unsigned reached_count, const float reach_percentage,
    const double avg_steps)
{
    std::cout << "Node: " << node_id << " reached " << reached_count << " times. ("
            << reach_percentage << ")%. avg step count: " << avg_steps << ".\n";
}

result_writer::result_writer(const std::string* path, const std::string* base_filename,
    const simulation_strategy strategy, const bool write_to_console, const bool write_to_file)
{
    this->path_ = *path;
    this->base_filename_ = *base_filename;
    this->strategy_ = strategy;
    this->write_to_console_ = write_to_console;
    this->write_to_file_ = write_to_file;
}

void result_writer::write_results(const simulation_result* sim_result, const unsigned result_size,
    const unsigned variable_count, steady_clock::duration sim_duration, const unsigned int i) const
{    
    const std::string filepath = this->path_ + this->base_filename_ + std::to_string(i) + ".txt";
    std::ofstream file;
    file.open(filepath);

    std::map<int, node_result> result_map;
    const array_t<variable_result> var_result = array_t<variable_result>(static_cast<int>(variable_count));

    for (unsigned int k = 0; k < static_cast<unsigned int>(var_result.size()); ++k)
    {
        var_result.arr()[k] = variable_result{k,0,0};
    }
    
    analyse_results(sim_result, strategy_.total_simulations(), &result_map, &var_result);

    file << "\naverage maximum value of each variable: \n";
    
    for (int j = 0; j < var_result.size(); ++j)
    {
        const variable_result* result = var_result.at(j);
        file << "variable " << result->variable_id << " = " << result->avg_max_value << "\n";
    }

    
    file << "\n\ngoal: \n";

    for (const std::pair<const int, node_result>& it : result_map)
    {
        if(it.first == HIT_MAX_STEPS) continue;
        const float percentage = calc_percentage(it.second.reach_count, result_size);
        print_node(it.first, it.second.reach_count, percentage, it.second.avg_steps);
    }

    const node_result no_value = result_map.at(HIT_MAX_STEPS);
    const float percentage = calc_percentage(no_value.reach_count, result_size);

    file << "No goal node was reached " << no_value.reach_count << " times. ("
         << percentage << ")%. avg step count: " << no_value.avg_steps << ".\n";

    file << "\nNr of simulations: " << result_size << "\n\n";

    file.flush();
    file.close();
}
