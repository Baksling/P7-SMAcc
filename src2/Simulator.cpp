#include "Simulator.h"
#include <iostream>

using namespace std;


static list<edge*> get_valid_edges(node* n)
{
    list<edge*> validated_data;
    
    for (edge* const it : n->get_edges()){
        if (it->validate())
        {
            validated_data.push_back(it);
        }
    }

    return validated_data;
}

static edge* choose_next_edge(const list<edge*>& edges)
{
    const unsigned int r = rand() % edges.size();  // NOLINT(concurrency-mt-unsafe)

    unsigned int i = 0;
    for (edge* const it : edges)
    {
        if (i != r)
        {
            i++;
            continue;
        }
        return it;
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
    const list<edge*> validated_data = get_valid_edges(start_node);

    if (validated_data.empty()) return false;

    //Find next route
    const edge* next_edge = choose_next_edge(validated_data);

    //Check if goal node hit // Else loop
    if (next_edge->get_node()->is_goal()) return true;
    return simulate(next_edge->get_node(), n_step + 1);
    
}


