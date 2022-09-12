// Your First C++ Program

#include <iostream>
#include <string>
#include "Node.h"
#include "Edge.h"
#include "RelationHandler.h"

using namespace std;

int main() {
    node node_one(1);
    node node_two(2);

    edge edge_one(&node_one, &node_two, 1);

    RelationHandler relation_handler;
    relation_handler.add_node(&node_one);
    relation_handler.add_edge_to_node(&node_one, &edge_one);
    relation_handler.add_edge_to_node(&node_two, &edge_one);

    const list<edge*> result = relation_handler.get_edges(&node_one);
    cout << result.front()->get_n2()->get_id();

    cout << edge_one.get_n1()->get_id();
    return 0;
}