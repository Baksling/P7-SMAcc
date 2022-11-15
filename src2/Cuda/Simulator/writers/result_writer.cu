#include "result_writer.h"

#include <cuda.h>
#include <iostream>
#include <fstream>

#include "../../UPPAALTreeParser/uppaal_tree_parser.h"

using namespace std::chrono;

float result_writer::calc_percentage(const unsigned long long counter, const unsigned long long divisor)
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
    const unsigned long long total_simulations,
    steady_clock::duration sim_duration)
{
    std::ofstream file_node, file_variable;
    
     file_node.open(this->file_path_ + "_node_data_" + std::to_string(output_counter_) + ".csv" );
     file_variable.open(this->file_path_ + "_variable_data_" + std::to_string(output_counter_) + ".csv");
    
     file_node << "Simulation,Model,Node,Steps,Time\n";
     file_variable << "Simulation,Variable,Value\n";
    
    
     for (unsigned i = 0; i < total_simulations; ++i)
     {
         const simulation_result local_result = results->meta_results[i];
         for (unsigned  j = 0; j < this->model_count_; ++j)
         {
             file_node << i << "," << j << "," << results->nodes[this->model_count_ * i + j] << "," << local_result.steps << "," << local_result.total_time_progress << "\n";
         }
    
         for (int k = 0; k < this->var_result_.size(); ++k)
         {
             file_variable << i << "," << k << "," << results->variables[this->var_result_.size() * i + k] << "\n";
         }
    }

    file_node.flush();
    file_variable.flush();
     
    file_node.close();
    file_variable.close();
}

void result_writer::write_summary_to_stream(std::ostream& stream, unsigned long long total_simulations,
                                            steady_clock::duration sim_duration) const
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

    stream << "No goal node was reached "<< no_value.reach_count << " times. ("
         << percentage << ")%. avg step count: " << no_value.avg_steps << ".\n";

    stream << "\nNr of simulations: " << total_simulations << "\n\n";
    stream << "Simulation ran for: " << duration_cast<milliseconds>(sim_duration).count() << "[ms]" << "\n";
}

void result_writer::write_trace(const result_manager* trace_tracker) const
{
    if(!(trace_tracker->tracking_trace && this->write_mode_ & trace)) return;

    const std::ofstream::openmode mode = std::ofstream::out | (this->output_counter_ == 0
                                                   ? std::ofstream::trunc
                                                   : std::ofstream::app);

    std::ofstream node_file = std::ofstream(this->file_path_ + "_trace_nodes.csv", mode);
    std::ofstream var_file = std::ofstream(this->file_path_ + "_trace_variables.csv",  mode);
    const trace_pointers trace = trace_tracker->load_trace();

    if(this->output_counter_ == 0)
    {
        node_file << "sim_id,step,node_id,time\n";
        var_file << "sim_id,step,var_id,value\n"; 
    }

    for (unsigned sim_id = 0; sim_id < trace.simulations; ++sim_id)
    {
        lend_array<trace_vector> vector = trace.get_trace(sim_id);
        for (int i = 0; i < vector.size(); ++i)
        {
            constexpr char c = ',';
            const trace_vector* v = vector.at(i);
            if(v->is_node)
                node_file << sim_id << c << v->step << c << v->item_id << c << v->value << '\n';
            else
                var_file << sim_id << c << v->step << c << v->item_id << c << v->value << '\n';
        }
    }

    node_file.flush();
    node_file.close();

    var_file.flush();
    var_file.close();

    trace.free_internals();
}

void result_writer::write_model(
    const std::unordered_map<int, std::string>* name_map,
    const std::unordered_map<int, node_with_system_id>* subsystem_map) const
{
    if(!(this->write_mode_ & trace || this->write_mode_ & model_out)) return;
    std::ofstream file = std::ofstream(this->file_path_ + "_model.csv",
        std::ofstream::out | std::ofstream::trunc);

    file << "node_id,subsystem_id,name,is_goal\n";

    for (auto pair : (*subsystem_map))
    {
        constexpr char c = ',';
        std::string name = name_map->count(pair.first) == 1
            ? name_map->at(pair.first)
              : std::to_string(pair.first);
        file << pair.first << c
            << pair.second.get_system_id() << c
            <<name << c
            << pair.second.get_node()->is_goal_node() << '\n';
    }

    file.flush();
    file.close();
}

void result_writer::clear()
{
    this->result_map_.clear();
    
    for (unsigned int k = 0; k < static_cast<unsigned int>(this->var_result_.size()); ++k)
    {
        this->var_result_.arr()[k] = variable_result{k,0,0};
    }
}


void result_writer::write_lite(const steady_clock::duration sim_duration) const
{
    std::ofstream lite;
    const std::string file_path = file_path_ + "_lite_summary.txt";
    lite.open(file_path);

    lite << duration_cast<milliseconds>(duration_cast<duration<double>>(sim_duration)).count();

    lite.flush();
    lite.close();
}

void result_writer::write_hit_file(const unsigned long long total_simulations) const
{
    std::ofstream file = std::ofstream(this->file_path_ + "_results.csv", std::ofstream::out|std::ofstream::trunc);
    
    if (result_map_.empty() || result_map_.size() == 1 && result_map_.count(HIT_MAX_STEPS))
    {
        file << "0";
        file.flush();
        file.close();
        return;
    } 
    for (const auto& pair : this->result_map_)
    {
        if (pair.first == HIT_MAX_STEPS) continue;

        const float percentage = this->calc_percentage(pair.second.reach_count, total_simulations);
        
        file << percentage << "\n";
    }  

    file.flush();
    file.close();
}


result_writer::result_writer(
    const std::string* path,
    const simulation_strategy strategy,
    const unsigned model_count,
    const unsigned variable_count,
    const int write_mode)
: trace_enabled(write_mode & trace)
{
    this->file_path_ = *path;
    this->strategy_ = strategy;
    this->write_mode_ = write_mode;
    this->model_count_ = model_count;
    this->output_counter_ = 0;
    this->result_map_ = std::unordered_map<int, node_result> ();
    this->var_result_ = array_t<variable_result>(static_cast<int>(variable_count));
    this->clear();
}

void result_writer::write(
    const result_manager* sim_result,
    const steady_clock::duration sim_duration)
{    
    if (this->write_mode_ & lite_sum)
    {
        this->write_lite(sim_duration);
    }

    if(this->write_mode_ & trace)
        this->write_trace(sim_result);

    if(this->write_mode_ & (console_sum | file_sum | file_data))
    {
        const sim_pointers pointers = sim_result->analyse(&this->result_map_, &this->var_result_);
        
        if(this->write_mode_ & file_data) write_to_file(&pointers, strategy_.total_simulations(), sim_duration);
        
        pointers.free_internals();
    }
    output_counter_++;
}

void result_writer::write_summary(const unsigned long long total_simulations, const steady_clock::duration sim_duration) const
{
    if (this->write_mode_ & console_sum)
    {
        this->write_summary_to_stream(std::cout, total_simulations, sim_duration);
    }
    if (this->write_mode_ & file_sum)
    {
        const std::string path = this->file_path_ + "_summary.csv";
    
        std::ofstream summary;
        summary.open(path);
        this->write_summary_to_stream(summary, total_simulations, sim_duration);

        summary.flush();
        summary.close();
    }
    if (this->write_mode_ & hit_file)
    {
        this->write_hit_file(total_simulations);   
    }
}

int result_writer::parse_mode(const std::string& str)
{
    if(str.empty()) return 0;

    int mode = 0;
    if(str.find('l') != std::string::npos) mode |= lite_sum;
    if(str.find('c') != std::string::npos) mode |= console_sum;
    if(str.find('f') != std::string::npos) mode |= file_sum;
    if(str.find('d') != std::string::npos) mode |= file_data;
    if(str.find('t') != std::string::npos) mode |= trace;
    if(str.find('m') != std::string::npos) mode |= model_out;
    if(str.find('p') != std::string::npos) mode |= pretty_out;
    if(str.find('r') != std::string::npos) mode |= hit_file;
    
    return mode;
}
