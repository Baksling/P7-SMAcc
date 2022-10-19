#include "UniformDistribution.h"

#include <stdlib.h>
#include <iostream>
#include <random>

double uniform_distribution::get_time_difference(const double difference)
{
    // double r_value = (static_cast<double> (rand()) / static_cast<double>(RAND_MAX)) * difference;   // NOLINT(concurrency-mt-unsafe)
    // std::cout << "ARGHHHHH: " << r_value << "\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, difference);
    return dis(gen);
    
}
