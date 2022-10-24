#pragma once
#include <map>
#include <string>
#include <chrono>

#include "simulation_result.h"
#include "simulation_strategy.h"
#include "../common/allocation_helper.h"
#include "../common/macro.h"
#include "../common/lend_array.h"

class result_writer
{
private:
    std::string path_;
    std::string base_filename_;
    simulation_strategy strategy_;
    bool write_to_file_;
    bool write_to_console_;
    unsigned file_count_ = 0;

    static void analyse_results(const simulation_result* simulation_results,
                                unsigned long total_simulations,
                                std::map<int, node_result>* results,
                                const array_t<variable_result>* avg_max_variable_value);

    static float calc_percentage(unsigned long counter, unsigned long divisor);

    static void print_node(int node_id, unsigned int reached_count, float reach_percentage, double avg_steps);

public:
    explicit result_writer(const std::string* path, const std::string* base_filename, simulation_strategy strategy,
        bool write_to_console = false, bool write_to_file = false);
    
    void write_results(const simulation_result* sim_result, const unsigned result_size, const unsigned variable_count,
        std::chrono::steady_clock::duration sim_duration) const;
};
