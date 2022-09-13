// Your First C++ Program

#include <iostream>
#include <string>
#include "Node.h"
#include "Edge.h"
#include "Simulator.h"

using namespace std;

int main() {
    node node_one(1);
    node node_two(2, false);

    node_one.add_edge(&node_two, 1);
    node_two.add_edge(&node_one, 1);

    simulator sim;
    sim.add_timer(1);

    node_two.add_guard(logical_operator::greater_equal, 0, sim.get_timer(1));
    
    cout << sim.simulate(&node_one);

    return 0;
}