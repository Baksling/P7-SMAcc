#pragma once
#include <array>
#include <string>
#include <map>
#include "result_store.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <unordered_map>

#include "../common/sim_config.h"

enum write_modes //values must be powers of 2.
{
    console_sum = 1,
    file_sum = 2,
    file_data = 4,
    lite_sum = 8,
    model_out = 32,
    pretty_out = 64,
    hit_file = 128
};

struct node_summary
{
    unsigned int reach_count = 0;
    double avg_steps = 0;
    double avg_time = 0;

    void add_reach(const unsigned int steps, const double time)
    {
        avg_steps = ((avg_steps * reach_count) + static_cast<double>(steps)) / (reach_count+1);
        avg_time = ((avg_time * reach_count) + time) / (reach_count+1);
        reach_count++;
    }

    void cumulative(const node_summary& other)
    {
        avg_steps = ((avg_steps * reach_count + other.avg_steps * other.reach_count)) / (reach_count + other.reach_count);
        reach_count += other.reach_count;
    }
};

struct variable_summary
{
    unsigned int variable_id;
    double avg_max_value;
    unsigned long values_counted;

    void update_count(const double max_value)
    {
        avg_max_value = ((avg_max_value * values_counted) + max_value) / (values_counted+1);
        values_counted++;
    }
};


struct output_properties
{
    std::unordered_map<int, std::string>* node_names;
    std::unordered_map<int, int>* node_network;
};


class output_writer
{
private:
    std::string file_path_;
    int write_mode_;
    unsigned output_counter_{};
    unsigned total_simulations_;
    unsigned model_count_;
    std::map<int, node_summary> node_summary_map_{};
    arr<variable_summary> variable_summaries_ = arr<variable_summary>::empty();
    double epsilon_;
    double alpha_;
    std::unordered_map<int, std::string> node_names_;
    std::unordered_map<int, int> node_network_;
    
    static float calc_percentage(const unsigned long long counter, const unsigned long long divisor);

    void write_to_file(const result_pointers* results,
        std::chrono::steady_clock::duration sim_duration) const;

    void write_summary_to_stream(std::ostream& stream, std::chrono::steady_clock::duration sim_duration) const;
    void write_lite(std::chrono::steady_clock::duration sim_duration) const;
    void write_hit_file(std::chrono::steady_clock::duration sim_duration) const;

public:
    explicit output_writer(const sim_config* config, const network* model);
    void write(const result_store* sim_result, std::chrono::steady_clock::duration sim_duration);
    void write_summary(std::chrono::steady_clock::duration sim_duration) const;
    void clear();
    static int parse_mode(const std::string& str);
};



inline int output_writer::parse_mode(const std::string& str)
{
    if(str.empty()) return 0;

    int mode = 0;
    if(str.find('l') != std::string::npos) mode |= lite_sum;
    if(str.find('c') != std::string::npos) mode |= console_sum;
    if(str.find('f') != std::string::npos) mode |= file_sum;
    if(str.find('d') != std::string::npos) mode |= file_data;
    if(str.find('m') != std::string::npos) mode |= model_out;
    if(str.find('p') != std::string::npos) mode |= pretty_out;
    if(str.find('r') != std::string::npos) mode |= hit_file;
    
    return mode;
}
