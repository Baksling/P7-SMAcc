#include "result_writer.h"

#include <cuda.h>
#include <iostream>
#include <fstream>

using namespace std::chrono;

// void result_writer::analyse_results(const simulation_result* simulation_results, const unsigned long total_simulations,
//     std::map<int, node_result>* results, const array_t<variable_result>* avg_max_variable_value, bool from_cuda) const
// {
//     const unsigned long long node_size = sizeof(int) * this->model_count_;
//
//     const cudaMemcpyKind kind = from_cuda ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
//     
//     double* local_variable_results = static_cast<double*>(malloc(sizeof(double) * avg_max_variable_value->size()));
//     int* local_node_results = static_cast<int*>(malloc(node_size));
//     
//     results->insert( std::pair<int, node_result>(HIT_MAX_STEPS, node_result{ 0, 0.0 }));
//     
//     for (unsigned long i = 0; i < total_simulations; ++i)
//     {
//         const simulation_result local_result = simulation_results[i];
//         
//         cudaMemcpy(local_node_results, local_result.end_node_id_arr, node_size, kind);
//         
//         for (unsigned int j = 0; j < this->model_count_ ; ++j)
//         {
//             if(results->count(local_node_results[j]) == 1)
//             {
//                 results->at(local_node_results[j]).update_count(local_result.steps);
//             }
//             else
//             {
//                 node_result r = {1, static_cast<double>(local_result.steps)};
//                 results->insert( std::pair<int, node_result>(local_node_results[j], r) );
//             }
//         }
//         
//         cudaMemcpy(local_variable_results, local_result.variables_max_value_arr, sizeof(double) * avg_max_variable_value->size(), kind);
//
//         for (int  j = 0; j < avg_max_variable_value->size(); ++j)
//         {
//             avg_max_variable_value->at(j)->update_count(local_variable_results[j]);
//         }
//     }
//     free(local_variable_results);
//     free(local_node_results);
// }

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

void result_writer::write_to_file(
    sim_pointers* results,
    std::unordered_map<int, node_result>* node_results,
    const array_t<variable_result>* var_result,
    unsigned long total_simulations,
    steady_clock::duration sim_duration) const
{        
    std::ofstream file_node, file_variable, summary;
    std::string path_to_write = this->file_path_ + "_summary.txt";
    summary.open(path_to_write);
    
    summary << "\naverage maximum value of each variable: \n";
    
    for (int j = 0; j < var_result->size(); ++j)
    {
        const variable_result* result = var_result->at(j);
        summary << "variable " << result->variable_id << " = " << result->avg_max_value << "\n";
    }
    
    summary << "\n\ngoal: \n";
    
    for (const std::pair<const int, node_result>& it : *node_results)
    {
        if(it.first == HIT_MAX_STEPS) continue;
        const float percentage = calc_percentage(it.second.reach_count, total_simulations);
        summary << print_node(it.first, it.second.reach_count, percentage, it.second.avg_steps);
    }
    
    const node_result no_value = node_results->at(HIT_MAX_STEPS);
    
    const float percentage = calc_percentage(no_value.reach_count, total_simulations);

    summary << "No goal node was reached " << no_value.reach_count << " times. ("
         << percentage << ")%. avg step count: " << no_value.avg_steps << ".\n";

    summary << "\nNr of simulations: " << total_simulations << "\n\n";

    duration<double> sim_duration_ = duration_cast<duration<double>>(sim_duration);
    summary << "Simulation ran for: " << duration_cast<milliseconds>(sim_duration_).count() << "[ms]" << "\n";

    summary.flush();
    summary.close();
    
     file_node.open(this->file_path_ + "_node_data.csv");
     file_variable.open(this->file_path_ + "_variable_data.csv");
    
     file_node << "Simulation,Model,Node,Steps,Time\n";
     file_variable << "Simulation,Variable,Value\n";
    
    
     for (unsigned i = 0; i < total_simulations; ++i)
     {

         
         const simulation_result local_result = results->meta_results[i];
        
         // cudaMemcpy(local_node_results, local_result.end_node_id_arr, node_size, kind);
         // cudaMemcpy(local_variable_results, local_result.variables_max_value_arr, sizeof(double) * var_result->size(), kind);
         //
         for (unsigned j = 0; j < this->model_count_; ++j)
         {
             file_node << i << "," << j << "," << results->nodes[this->model_count_ * i + j] << "," << local_result.steps << "," << local_result.total_time_progress << "\n";
         }
    
         for (unsigned k = 0; k < this->variable_count_; ++k)
         {
             file_variable << i << "," << k << "," << results->variables[this->variable_count_ * i + k] << "\n";
         }
    }

    // free(local_node_results);
    // free(local_variable_results);
    
    file_node.flush();
    file_variable.flush();
     
    file_node.close();
    file_variable.close();
}

void result_writer::write_to_console(
    const std::unordered_map<int, node_result>* results,
    const unsigned long total_simulations,
    const array_t<variable_result> var_result) const
{
    std::cout << this->file_path_;
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

void result_writer::write_lite(std::chrono::steady_clock::duration sim_duration) const
{
    std::ofstream lite;
    std::string file_path = file_path_ + "_lite_summary.txt";
    lite.open(file_path);

    lite << duration_cast<milliseconds>(duration_cast<duration<double>>(sim_duration)).count();

    lite.flush();
    lite.close();
}


result_writer::result_writer(const std::string* path,const simulation_strategy strategy, const unsigned model_count,
                             const unsigned variable_count, const unsigned write_mode)
{
    this->file_path_ = *path;
    this->strategy_ = strategy;
    this->write_mode_ = write_mode;
    this->model_count_ = model_count;
    this->variable_count_ = variable_count;
    
}

void result_writer::write_results(
    const simulation_result_container* sim_result,
    steady_clock::duration sim_duration) const
{
    std::unordered_map<int, node_result> result_map;
    const array_t<variable_result> var_result = array_t<variable_result>(static_cast<int>(this->variable_count_));

    
    for (unsigned int k = 0; k < static_cast<unsigned int>(var_result.size()); ++k)
    {
        var_result.arr()[k] = variable_result{k,0,0};
    }

    if (this->write_mode_ == 3)
    {
        this->write_lite(sim_duration);
        return;
    }
    
    sim_pointers pointers = sim_result->analyse(&result_map, &var_result);
    
    if (this->write_mode_ == 0 || this->write_mode_ == 2) write_to_file(&pointers, &result_map, &var_result,
                                            strategy_.total_simulations(), sim_duration);

    if (this->write_mode_ == 1 || this->write_mode_ == 2) write_to_console(&result_map, strategy_.total_simulations(), var_result);

    pointers.free_internals();
}
