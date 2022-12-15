#include "output_writer.h"

float output_writer::calc_percentage(const unsigned long long counter, const unsigned long long divisor)
{
    return (static_cast<float>(counter) / static_cast<float>(divisor)) * 100.0f;
}

void output_writer::write_to_file(const result_pointers* results,
    std::chrono::steady_clock::duration sim_duration) const
{
    std::ofstream file_node, file_variable;
    
    file_node.open(this->file_path_ + "_node_data_" + std::to_string(output_counter_) + ".csv" );
    file_variable.open(this->file_path_ + "_variable_data_" + std::to_string(output_counter_) + ".csv");
    
    file_node << "Simulation,Model,Node,Steps,Time\n";
    file_variable << "Simulation,Variable,Value\n";
    
    
    for (unsigned i = 0; i < total_simulations_; ++i)
    {
        const sim_metadata local_result = results->meta_results[i];
        for (unsigned  j = 0; j < this->model_count_; ++j)
        {
            file_node << i << "," << j << "," << results->nodes[this->model_count_ * i + j] << ","
            << local_result.steps << "," << local_result.global_time << "\n";
        }
    
        for (int k = 0; k < this->variable_summaries_.size; ++k)
        {
            file_variable << i << "," << k << "," << results->variables[this->variable_summaries_.size * i + k] << "\n";
        }
    }

    file_node.flush();
    file_variable.flush();
     
    file_node.close();
    file_variable.close();
}

void insert_name(const int id, const std::unordered_map<int, std::string>& name_map, std::ostream& stream)
{
    stream << abs(id) << ": ";
    if(name_map.count(abs(id)))
    {
        stream << name_map.at(abs(id));
    }
    else
    {
        stream << '_';
    }
    stream << (HAS_HIT_MAX_STEPS(id) ? " (not goal)" : "");
}

void output_writer::write_summary_to_stream(std::ostream& stream,
                                            std::chrono::steady_clock::duration sim_duration) const
{
    stream << "\nAverage maximum value of each variable: \n";
    
    for (int j = 0; j < this->variable_summaries_.size; ++j)
    {
        const variable_summary result = this->variable_summaries_.store[j];
        stream << "variable " << result.variable_id << " = " << result.avg_max_value << "\n";
    }
    
    stream << "\n\nReachability: \n";
    bool has_seen_hit = false;
    
    // node_summary no_hit = node_summary{0, 0};
    std::map<int, node_summary> no_hit_map{};
    stream << "Not reached:\n";
    
    for (const std::pair<const int, node_summary>& it : this->node_summary_map_)
    {
        if(HAS_HIT_MAX_STEPS(it.first))
        {
            // no_hit.cumulative(it.second);
            int network = this->node_network_.count(abs(it.first))
            ? this->node_network_.at(abs(it.first))
            : 0;
            
            if(no_hit_map.count(network))
                no_hit_map[network].cumulative(it.second);
            else
                no_hit_map[network] = it.second;
        }
        else  if(!has_seen_hit)
        {
            has_seen_hit = true;
            stream << "Goals: \n";
        }
        
        const float percentage = calc_percentage(it.second.reach_count, total_simulations_);
        insert_name(it.first, this->node_names_, stream);
        stream << " reached " << it.second.reach_count << " times (" << percentage << "%)\n";
        stream << "    " << " avg steps: " << it.second.avg_steps << " | avg sim time: " << it.second.avg_time << "t.\n";
    }

    stream << "\nPer process\n";
    node_summary global_no_hit = {0,0,0};
    for (int i = 0; i < static_cast<int>(this->model_count_); ++i)
    {
        const node_summary no_hit = no_hit_map.count(i)
                                        ? no_hit_map[i]
                                        : node_summary{0, 0, 0} ;
        global_no_hit.cumulative(no_hit);
        const float percentage = calc_percentage(no_hit.reach_count, total_simulations_);

        stream << "Process " << i << " did not reach goal " << no_hit.reach_count << " times ("<< percentage <<"%)\n";
        stream << "    " <<" avg steps: " << no_hit.avg_steps << " | avg sim time: " << no_hit.avg_time << "t.\n";
    }

    stream << '\n';
    stream << "Probability of false negative results (alpha): " << this->alpha_*100 << "%\n";  
    stream << "Probability uncertainty (-+epsilon) of results: " << this->epsilon_*100 << "%\n";  
    stream << "Nr of simulations: " << total_simulations_ << "\n\n";
    stream << "Simulation ran for: " << std::chrono::duration_cast<std::chrono::milliseconds>(sim_duration).count()
           << "[ms]" << "\n";
}

void output_writer::write_lite(std::chrono::steady_clock::duration sim_duration) const
{
    std::ofstream lite;
    const std::string file_path = file_path_ + "_lite_summary.txt";
    lite.open(file_path);

    lite << std::chrono::duration_cast<std::chrono::milliseconds>(sim_duration).count();

    lite.flush();
    lite.close();
}

void output_writer::write_hit_file(const std::chrono::steady_clock::duration sim_duration) const
{
    if (this->node_summary_map_.empty()) return;
    std::ofstream file = std::ofstream(this->file_path_ + "_results.tsv", std::ofstream::out|std::ofstream::trunc);
    
    bool any = false;
    for (const auto& pair : this->node_summary_map_)
    {
        if (HAS_HIT_MAX_STEPS(pair.first)) continue;
        const float percentage = this->calc_percentage(pair.second.reach_count, total_simulations_);
        file << percentage << "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(sim_duration).count();
        any = true;
    }
    if(!any)
    {
        file << "0.0\t" << std::chrono::duration_cast<std::chrono::milliseconds>(sim_duration).count();
    }

    file.flush();
    file.close();
}

output_writer::output_writer(const sim_config* config, const network* model)
{
    this->file_path_ = config->paths->output_path;
    this->write_mode_ = config->write_mode;
    this->node_summary_map_ = std::map<int, node_summary>();
    this->model_count_ = config->network_size;
    this->total_simulations_ = static_cast<unsigned>(config->total_simulations()) * config->simulation_repetitions;
    this->alpha_ = config->alpha;
    this->epsilon_ = config->epsilon;
    this->node_names_ = std::unordered_map<int, std::string>(*config->properties->node_names);
    this->node_network_ = std::unordered_map<int, int>(*config->properties->node_network);
    
    this->variable_summaries_ = arr<variable_summary>{
        static_cast<variable_summary*>(malloc(sizeof(variable_summary)*config->tracked_variable_count)),
        static_cast<int>(config->tracked_variable_count)
    };
    
    for (int i = 0, j = 0; i < model->variables.size; ++i)
    {
        if(!model->variables.store[i].should_track) continue;
        this->variable_summaries_.store[j++] = variable_summary{static_cast<unsigned>(i), 0.0, 0};
    }

}

void output_writer::write(const result_store* sim_result, std::chrono::steady_clock::duration sim_duration)
{
    if (this->write_mode_ & lite_sum)
    {
        this->write_lite(sim_duration);
    }

    if(this->write_mode_ & (console_sum | file_sum | file_data | hit_file))
    {
        const result_pointers pointers = sim_result->load_results();

        for (unsigned i = 0; i < this->total_simulations_; ++i)
        {
            const sim_metadata x = pointers.meta_results[i];
            for (unsigned j = 0; j < model_count_; ++j)
            {
                int n = pointers.nodes[i*model_count_ + j];
                if(this->node_summary_map_.count(n)) //exists
                    this->node_summary_map_[n].add_reach(x.steps, x.global_time);
                else
                    this->node_summary_map_.insert(
                        std::pair<int, node_summary>(
                            n, node_summary{ 1, static_cast<double>(x.steps), x.global_time }
                            ));
            }
            for (int j = 0; j < this->variable_summaries_.size; ++j)
            {
                const double v = pointers.variables[i*this->variable_summaries_.size + j];
                this->variable_summaries_.store[j].update_count(v);
            }
        }
        if(this->write_mode_ & file_data) write_to_file(&pointers, sim_duration);
        pointers.free_internals();
    }
    output_counter_++;
}

void output_writer::write_summary(std::chrono::steady_clock::duration sim_duration) const
{
    if (this->write_mode_ & console_sum)
    {
        this->write_summary_to_stream(std::cout, sim_duration);
    }
    if (this->write_mode_ & file_sum)
    {
        const std::string path = this->file_path_ + "_summary.csv";
    
        std::ofstream summary;
        summary.open(path);
        this->write_summary_to_stream(summary, sim_duration);

        summary.flush();
        summary.close();
    }
    if (this->write_mode_ & hit_file)
    {
        this->write_hit_file(sim_duration);   
    }
}

void output_writer::clear()
{
    this->node_summary_map_.clear();
    for (int i = 0; i < this->variable_summaries_.size; ++i)
    {
        this->variable_summaries_.store[i] = variable_summary{
            this->variable_summaries_.store[i].variable_id,
            0.0,
            0,
        };
    }
}
