#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "Node.h"
#include "Edge.h"
#include "Timer.h"
#include <map>

class simulator
{
private:
    int max_steps_;
    map<int, timer*> timers_;
    double find_least_difference(node* current_node);
    node* simulate_process(node* current_node);
public:
    simulator(int max_steps = 1000);
    map<node*, unsigned int>* simulate(node* start_node, int number_of_simulations = 100);
    void add_timer(int id);
    timer* get_timer(int id);
    void update_time(node* current_node);
};

#endif // SIMULATOR_H
