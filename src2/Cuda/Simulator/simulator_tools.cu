#include "simulator_tools.h"


CPU GPU bool bit_handler::bit_is_set(const unsigned long long* n, const unsigned int i)
{
    return (*n) & (1UL << i);
}


CPU GPU void bit_handler::set_bit(unsigned long long* n, const unsigned int i)
{
    (*n) |=  (1UL << (i));
}

CPU GPU void bit_handler::unset_bit(unsigned long long* n, const unsigned int i)
{
    (*n) &= ~(1UL << (i));
}


void result_handler::read_results(const int* cuda_results, const unsigned long total_simulations, std::map<int, unsigned long>* results)
{
    for (unsigned long i = 0; i < total_simulations; ++i)
    {
        int id = cuda_results[i];
        unsigned long count = 0;
        if(results->count(id) == 1)
        {
            count  = results->at(id);
        }

        results->insert_or_assign(id, count+1);

    }
}

float result_handler::calc_percentage(const unsigned long counter, const unsigned long divisor)
{
    return (static_cast<float>(counter)/static_cast<float>(divisor))*100;
} 

void result_handler::print_results(std::map<int,unsigned long>* result_map, const unsigned long result_size)
{
    std::cout << "\n";
    for (const std::pair<const int, int> it : (*result_map))
    {
        if(it.first == HIT_MAX_STEPS) continue;
        const float percentage = calc_percentage(it.second, result_size);
        std::cout << "Node: " << it.first << " reached " << it.second << " times. (" << percentage << ")%\n";
    }
    const float percentage = calc_percentage((*result_map)[HIT_MAX_STEPS], result_size);
    std::cout << "No goal state was reached " << (*result_map)[HIT_MAX_STEPS] << " times. (" << percentage << ")%\n";
    std::cout << "\n";
    std::cout << "Nr of simulations: " << result_size << "\n";
    std::cout << "\n";

}