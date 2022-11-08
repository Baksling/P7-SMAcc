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
    const sim_pointers* results,
    unsigned long long total_simulations,
    steady_clock::duration sim_duration)
{
    std::ofstream file_node, file_variable;
    
     file_node.open(this->file_path_ + "_node_data_" + std::to_string(simulation_counter_) + ".csv" );
     file_variable.open(this->file_path_ + "_variable_data_" + std::to_string(simulation_counter_) + ".csv");

     simulation_counter_++;
    
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
    
         for (unsigned k = 0; k < this->var_result_.size(); ++k)
         {
             file_variable << i << "," << k << "," << results->variables[this->var_result_.size() * i + k] << "\n";
         }
    }

    // free(local_node_results);
    // free(local_variable_results);
    
    file_node.flush();
    file_variable.flush();
     
    file_node.close();
    file_variable.close();
}

void result_writer::write_summary_to_stream(std::ostream& stream, unsigned long long total_simulations, std::chrono::steady_clock::duration sim_duration) const
{
    stream << "\naverage maximum value of each variable: \n";
    
    for (int j = 0; j < this->var_result_.size(); ++j)
    {
        const variable_result* result = this->var_result_.at(j);
        stream << "variable " << result->variable_id << " = " << result->avg_max_value << "\n";
    }
    
    stream << "\n\ngoal: \n";
    
    for (const std::pair<const int, node_result>& it : this->result_map_)
    {
        if(it.first == HIT_MAX_STEPS) continue;
        const float percentage = calc_percentage(it.second.reach_count, total_simulations);
        stream << print_node(it.first, it.second.reach_count, percentage, it.second.avg_steps);
    }
    
    const node_result no_value = this->result_map_.at(HIT_MAX_STEPS);
    
    const float percentage = calc_percentage(no_value.reach_count, total_simulations);

    stream << "No goal node was reached " << no_value.reach_count << " times. ("
         << percentage << ")%. avg step count: " << no_value.avg_steps << ".\n";

    stream << "\nNr of simulations: " << total_simulations << "\n\n";

    duration<double> sim_duration_ = duration_cast<duration<double>>(sim_duration);
    stream << "Simulation ran for: " << duration_cast<milliseconds>(sim_duration_).count() << "[ms]" << "\n";
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
                             const unsigned variable_count, writer_modes write_mode)
{
    this->file_path_ = *path;
    this->strategy_ = strategy;
    this->write_mode_ = write_mode;
    this->model_count_ = model_count;
    this->simulation_counter_ = 0;
    this->result_map_ = std::unordered_map<int, node_result> ();
    this->var_result_ = array_t<variable_result>(static_cast<int>(variable_count));

    for (unsigned int k = 0; k < static_cast<unsigned int>(this->var_result_.size()); ++k)
    {
        this->var_result_.arr()[k] = variable_result{k,0,0};
    }
}

void result_writer::write_results(
    const simulation_result_container* sim_result,
    steady_clock::duration sim_duration)
{    
    if (this->write_mode_ == writer_modes::lite_sum)
    {
        this->write_lite(sim_duration);
        return;
    }

    if (this->write_mode_ == writer_modes::no_output) return;
    
    const sim_pointers pointers = sim_result->analyse(&this->result_map_, &this->var_result_);
    
    if (this->write_mode_  == writer_modes::console_file_data
        || this->write_mode_ == writer_modes::console_sum_file_sum_file_data
        || this->write_mode_  == writer_modes::file_sum_file_data)
    {
        write_to_file(&pointers, strategy_.total_simulations(), sim_duration);
    }

    pointers.free_internals();
}

void result_writer::write_summary(unsigned long long total_simulations, std::chrono::steady_clock::duration sim_duration)
{
    if (this->write_mode_ == writer_modes::no_output || this->write_mode_ == writer_modes::lite_sum) return;
    if (this->write_mode_ == writer_modes::console_sum
        || this->write_mode_ == writer_modes::console_file_sum
        || this->write_mode_ == writer_modes::console_file_data
        || this->write_mode_ == writer_modes::console_sum_file_sum_file_data)
    {
        this->write_summary_to_stream(std::cout, total_simulations, sim_duration);
    }
    if (this->write_mode_ == writer_modes::console_sum_file_sum_file_data
        || this->write_mode_ == writer_modes::file_sum
        || this->write_mode_ == writer_modes::file_sum_file_data    
        || this->write_mode_ == writer_modes::console_file_sum)
    {
        std::string path = this->file_path_ + "_summary.csv";
    
        std::ofstream summary;
        summary.open(path);
        this->write_summary_to_stream(summary, total_simulations, sim_duration);

        summary.flush();
        summary.close();
    }

}
