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
    std::string file_path_;
    simulation_strategy strategy_;
    bool write_to_file_;
    bool write_to_console_;
    unsigned model_count_;
    unsigned variable_count_;

    static float calc_percentage(unsigned long counter, unsigned long divisor);

    static std::string print_node(int node_id, unsigned int reached_count, float reach_percentage, double avg_steps);

    void write_to_file(sim_pointers* results, std::unordered_map<int, node_result>* node_results,
                       const array_t<variable_result>* var_result,
                       unsigned long total_simulations, std::chrono::steady_clock::duration sim_duration) const;

    void write_to_console(const std::unordered_map<int, node_result>* results,
                          const unsigned long total_simulations, const array_t<variable_result> var_result) const;

public:
    explicit result_writer(const std::string* path, simulation_strategy strategy, unsigned model_count,
        unsigned variable_count, bool write_to_console = false, bool write_to_file = false);
    
    void write_results(const simulation_result_container* sim_result,
                       std::chrono::steady_clock::duration sim_duration) const;
};
