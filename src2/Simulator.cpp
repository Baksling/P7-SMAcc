#include "Simulator.h"
#include <iostream>
#include <map>
#include "UniformDistribution.h"
#include <cfloat>

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


bool simulator::simulate(node* start_node, int n_step)
{
    node* current_node = start_node;
    int current_step = 0;
    do
    {
        current_step ++;
        this->update_time(start_node);
    
        //Get validated edges! (Active edges)
        list<edge>* validated_data = get_valid_edges(current_node);

        if (validated_data->empty()) continue;
    
        //Find next route
        const edge* next_edge = choose_next_edge(validated_data);

        delete validated_data;
    
        //Check if goal node hit // Else loop
        if (next_edge->get_node()->is_goal()) return true;
        current_node = next_edge->get_node();
        
    } while (current_step <= this->max_steps_);

    cout << "Hit max steps!\n";
    return false;
    
}

void simulator::add_timer(const int id)
{
    this->timers_[id] = new timer();
}

timer* simulator::get_timer(const int id)
{
    return this->timers_[id];
}

void simulator::update_time(node* current_node)
{
    double least_difference = this->find_least_difference(current_node);
    
    uniform_distribution tmp;
    double time_progression = tmp.get_time_difference(least_difference);

    for (auto const& t : this->timers_)
    {
        t.second->set_time(t.second->get_time() + time_progression);
        cout << t.first << ": " << t.second->get_time() << "\n";
    }

    
}

double simulator::find_least_difference(node* current_node)
{
    double least_difference = DBL_MAX;
    for (auto const& t : this->timers_)
    {
        for (guard g : current_node->get_invariants())
        {
            if (g.get_type() != logical_operator::less_equal &&
                g.get_type() != logical_operator::less) continue;
            
            double diff = g.get_value() - t.second->get_time();
            if (diff >= 0 && diff < least_difference)
                least_difference = diff;
        }
    }

    return least_difference;
}

