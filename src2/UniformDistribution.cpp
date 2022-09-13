#include "UniformDistribution.h"

#include <stdlib.h>

double uniform_distribution::get_time_difference(const double difference)
{
    return static_cast<double> (rand()) / static_cast<double>(RAND_MAX) * difference;   // NOLINT(concurrency-mt-unsafe)
}
