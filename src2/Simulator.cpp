#include "Simulator.h"
#include <iostream>
#include <map>

using namespace std;


static list<edge>* get_valid_edges(node* n)
{
    list<edge>* validated_data = new list<edge>;
    
    for (edge it : n->get_edges()){
        if (it.validate())
        {
            validated_data->push_back(it);
        }
    }

    return validated_data;
}

static edge* choose_next_edge(list<edge>* edges)
{
    const unsigned int r = rand() % edges->size();  // NOLINT(concurrency-mt-unsafe)

    unsigned int i = 0;
    for (edge& it : *edges)
    {
        if (i != r)
        {
            i++;
            continue;
        }
        return &it;
    }
    return nullptr;
}

simulator::simulator(const int max_steps)
{
    this->max_steps_ = max_steps;
}


bool simulator::simulate(node* start_node, const int n_step)
{
    if (n_step >= this->max_steps_)
    {
        cout << "Hit max steps!";
        return false;
    }
    
    //Get validated edges! (Active edges)
    list<edge>* validated_data = get_valid_edges(start_node);

    if (validated_data->empty()) return false;

    //Find next route
    const edge* next_edge = choose_next_edge(validated_data);

    delete validated_data;
    
    //Check if goal node hit // Else loop
    if (next_edge->get_node()->is_goal()) return true;
    return simulate(next_edge->get_node(), n_step + 1);
    
}

void simulator::add_timer(const int id)
{
    this->timers_[id] = new timer();
}

timer* simulator::get_timer(const int id)
{
    return this->timers_[id];
}



