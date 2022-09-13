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
    cout << sim.simulate(&node_one);

    return 0;
}