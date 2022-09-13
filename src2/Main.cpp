// Your First C++ Program

#include <iostream>
#include <string>
#include "Node.h"
#include "Edge.h"
#include "Simulator.h"

using namespace std;

int main() {
    node node_one(1);
    node node_two(2, true);

    simulator sim;
    sim.add_timer(1);
    sim.add_timer(2);
    
    list<guard> hej;
    hej.emplace_back(logical_operator::greater_equal, 2, sim.get_timer(2));

    node_one.add_edge(&node_two, hej);
    //node_two.add_edge(&node_one, );

    node_one.add_invariant(logical_operator::less_equal, 1, sim.get_timer(1));
    node_two.add_invariant(logical_operator::greater_equal, 1, sim.get_timer(1));
    node_two.add_invariant(logical_operator::greater_equal, 1, sim.get_timer(2));

    bool result = sim.simulate(&node_one);
    
    cout << "Result: " << result << "\n";

    return 0;
}