#include "Simulator.h"
#include <iostream>
#include <map>
#include "UniformDistribution.h"
#include <cfloat>

using namespace std;


static list<edge>* get_valid_edges(node* n)
{
    list<edge>* validated_data = new list<edge>;
    
    for (edge it : *(n->get_edges())){
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


map<node*, unsigned int>* simulator::simulate(node* start_node, int number_of_simulations)
{
    map<node*, unsigned int>* result_map = new map<node*, unsigned>;

    for (int i = 0; i < number_of_simulations; i++)
    {
        for (auto const& t : this->timers_)
        {
            t.second->set_time(0);
        }
        
        if (i % 10000 == 0) cout << "Simulations started: " << i << "\n";
        node* result_node = this->simulate_process(start_node);
        
        if (!result_node->is_goal()) continue;
        
        map<node*, unsigned int>::iterator pos = result_map->find(result_node);
        if (pos == result_map->end())
        {
            result_map->insert(pair<node*, unsigned int>(result_node, 1));
        }
        else
        {
            pos->second++;
        }
        
    }

    return result_map;
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
    }
}

double simulator::find_least_difference(node* current_node)
{
    double least_difference = DBL_MAX;
    for (auto const& t : this->timers_)
    {
        for (guard g : *(current_node->get_invariants()))
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

node* simulator::simulate_process(node* start_node)
{
    node* current_node = start_node;
    int current_step = 0;
    do
    {
        this->update_time(start_node);
        current_step ++;

        // for (auto const& t : this->timers_)
        // {
        //     cout << t.first << ": " << t.second->get_time() << "\n";
        // }
        
        //Check if current node has edges (Otherwise it terminates if not an goal)
        if (current_node->get_edges()->empty() || current_node->is_goal())
            return current_node;
        
        //Get validated edges! (Active edges)
        list<edge>* validated_data = get_valid_edges(current_node);
        
        if (validated_data->empty()) continue;
    
        //Find next route
        const edge* next_edge = choose_next_edge(validated_data);

        delete validated_data;
    
        //Check if goal node hit // Else loop
        current_node = next_edge->get_node();
        if (current_node->is_goal()) return current_node;
        
        
        
    } while (current_step <= this->max_steps_);

    cout << "Hit max steps!\n";
    return current_node;
}

