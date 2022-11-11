#pragma once
#include <map>
#include <string>
#include <chrono>

#include "simulation_result.h"
#include "simulation_strategy.h"
#include "../common/allocation_helper.h"
#include "../common/macro.h"
#include "../common/lend_array.h"

enum writer_modes
{
    no_output = 0,
    console_sum = 1,
    file_sum = 2,
    console_file_sum = 3,
    console_file_data = 4,
    file_sum_file_data = 5,
    console_sum_file_sum_file_data = 6,
    lite_sum = 7
};

class result_writer
{
private:
    std::string file_path_;
    simulation_strategy strategy_;
    writer_modes write_mode_;
    unsigned model_count_;
    unsigned simulation_counter_;
    std::unordered_map<int, node_result> result_map_;
    array_t<variable_result> var_result_{0};

    static float calc_percentage(const unsigned long long counter, const unsigned long long divisor);

    static std::string print_node(int node_id, unsigned int reached_count, float reach_percentage, double avg_steps);

    void write_to_file(const sim_pointers* results,
                       unsigned long long total_simulations, std::chrono::steady_clock::duration sim_duration);

    void write_to_console(const unsigned long total_simulations) const;
    void write_summary_to_stream(std::ostream& stream, unsigned long long total_simulations, std::chrono::steady_clock::duration sim_duration) const;

    void write_lite(std::chrono::steady_clock::duration sim_duration) const;

public:
    explicit result_writer(const std::string* path, simulation_strategy strategy, unsigned model_count,
        unsigned variable_count, writer_modes write_mode);
    
    void write_results(const simulation_result_container* sim_result,
                       std::chrono::steady_clock::duration sim_duration);

    void write_summary(unsigned long long total_simulations, std::chrono::steady_clock::duration sim_duration);

    void clear();
    
};
