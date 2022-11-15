#pragma once

struct node_result
{
    unsigned int reach_count;
    double avg_steps;

    void update_count(const unsigned int avg_step)
    {
        avg_steps = ((avg_steps * reach_count) + static_cast<double>(avg_step)) / (reach_count+1);
        reach_count++;
    }
};

struct variable_result
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

struct simulation_result
{
    unsigned int steps;
    double total_time_progress;
};
