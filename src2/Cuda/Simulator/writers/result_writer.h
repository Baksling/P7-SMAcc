#pragma once
#include <unordered_map>
#include <string>
#include <chrono>

#include "result_manager.h"

#include "simulation_result.h"
#include "../simulation_strategy.h"
#include "../../common/macro.h"


//! MUST BE POWERS OF 2 AS IT IS USEd IN BITWISE OPERATIONS
//Write summary console,
//Write summary file
//write file data
//Write trace
//write lite
enum writer_modes
{
    console_sum = 1,
    file_sum = 2,
    file_data = 4,
    trace = 8,
    lite_sum = 16
};

class result_writer
{
private:
    std::string file_path_;
    simulation_strategy strategy_;
    int write_mode_;
    unsigned model_count_;
    unsigned output_counter_;
    std::unordered_map<int, node_result> result_map_{};
    array_t<variable_result> var_result_{0};

    static float calc_percentage(const unsigned long long counter, const unsigned long long divisor);

    static std::string print_node(int node_id, unsigned int reached_count, float reach_percentage, double avg_steps);

    void write_to_file(const sim_pointers* results,
                       unsigned long long total_simulations, std::chrono::steady_clock::duration sim_duration);

    void write_summary_to_stream(std::ostream& stream, unsigned long long total_simulations, std::chrono::steady_clock::duration sim_duration) const;

    void write_trace(const result_manager* trace_tracker) const;
    
    void write_lite(std::chrono::steady_clock::duration sim_duration) const;

public:
    const bool trace_enabled;

    explicit result_writer(const std::string* path, simulation_strategy strategy, unsigned model_count,
        unsigned variable_count, int write_mode);
    
    void write(const result_manager* sim_result,
                       std::chrono::steady_clock::duration sim_duration);

    void write_summary(unsigned long long total_simulations, std::chrono::steady_clock::duration sim_duration) const;

    void write_model(const stochastic_model_t* model,
                     std::unordered_map<int, std::string>& name_map,
                     std::unordered_map<int, int>& model_map);
    void clear();
    
    static int parse_mode(const std::string& str);
};
