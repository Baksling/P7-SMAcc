#ifndef SIMULATOR_TOOLS_H
#define SIMULATOR_TOOLS_H

#include "../common/macro.h"
#include "../common/lend_array.h"
#include <map>
#include "simulation_result.h"
#include "simulation_strategy.h"
#include "../common/allocation_helper.h"


class bit_handler
{
public:
    CPU GPU static bool bit_is_set(const unsigned long long* n, unsigned int i);
    CPU GPU static void set_bit(unsigned long long* n, unsigned int i);
    CPU GPU static void unset_bit(unsigned long long* n, unsigned int i);  
};


class result_handler
{
public:
    static array_t<variable_result> allocate_variable_results(const unsigned variable_count);
    static simulation_result* allocate_results(const simulation_strategy* strategy,
                                               const unsigned variable_count, const allocation_helper* helper, bool cuda_allocate);
    static void read_results(const simulation_result* simulation_results, unsigned long total_simulations,
    std::map<int, node_result>* results, const lend_array<variable_result>* avg_max_variable_value);
    static void print_results(std::map<int, node_result>* result_map, const lend_array<variable_result>* variable_results, const unsigned long
                              result_size);
};


#endif