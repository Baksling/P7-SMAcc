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

    void analyse_results(const simulation_result* simulation_results,
                            const unsigned long total_simulations,
                            std::map<int, node_result>* results,
                            const array_t<variable_result>* avg_max_variable_value, bool from_cuda) const;

    static float calc_percentage(unsigned long counter, unsigned long divisor);

    static std::string print_node(int node_id, unsigned int reached_count, float reach_percentage, double avg_steps);

    void write_to_file(const simulation_result* sim_result, std::map<int, node_result>* results,
                       unsigned long total_simulations, array_t<variable_result> var_result, unsigned variable_count) const;

    void write_to_console(std::map<int, node_result>* results,
        unsigned long total_simulations, array_t<variable_result> var_result) const;

public:
    explicit result_writer(const std::string* path, simulation_strategy strategy, unsigned model_count,
        bool write_to_console = false, bool write_to_file = false);
    
    void write_results(const simulation_result* sim_result, const unsigned result_size, const unsigned variable_count,
                       std::chrono::steady_clock::duration sim_duration, bool from_cuda) const;
};
