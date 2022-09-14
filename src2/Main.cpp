// Your First C++ Program

#include <iostream>
#include <string>
#include "Node.h"
#include "Edge.h"
#include "Simulator.h"
#include "Update.h"
#include <map>
#include <time.h>

using namespace std;

static void print_result(map<node*, unsigned int>* result, const int number_of_simulations)
{
    for (map<node*, unsigned int>::iterator it = result->begin(); it != result->end(); it++)
    {
        cout << it->first->get_id() << ": " << it->second << " (~" << static_cast<int>(static_cast<double>(it->second) / static_cast<double>(number_of_simulations) * 100) << "%)\n";
    }
    cout << "Number of simulations: " << number_of_simulations << "\n";
}

int main() {
    srand(time(NULL));
    const int number_of_simulations = 1000;

    cout << "Hej";
    

    // Clock initialization
    simulator sim;
    sim.add_timer(1);


    // Node initialization
    node node_one(1);
    node node_two(2, true);
    node node_three(3);

    //Invariant initialization
    node_one.add_invariant(logical_operator::less_equal, 10, sim.get_timer(1));

    //Edge guard initialization
    list<guard> edge12_guard;
    edge12_guard.emplace_back(logical_operator::less_equal, 10, sim.get_timer(1));

    list<guard> edge13_guard;
    edge13_guard.emplace_back(logical_operator::less_equal, 3, sim.get_timer(1));

    //Update initialization
    list<update>* edge12_update = new list<update>;
    edge12_update->emplace_back(sim.get_timer(1), 0);

    list<update>* edge13_update = new list<update>;
    edge13_update->emplace_back(sim.get_timer(1), 100);

    //Edge initialization
    node_one.add_edge(&node_two, edge12_guard, edge12_update);
    node_one.add_edge(&node_three, edge13_guard, edge13_update);
    
    map<node*, unsigned int>* result = sim.simulate(&node_one, number_of_simulations);

    print_result(result, number_of_simulations);
    
    return 0;
}