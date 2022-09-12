// Your First C++ Program

#include <iostream>
#include <string>
#include "Node.h"
#include "Edge.h"
#include "RelationHandler.h"

using namespace std;

int main() {
    Node nodeOne(1);
    Node nodeTwo(2);

    Edge edgeOne(&nodeOne, &nodeTwo, 1);

    RelationHandler relationHandler;
    relationHandler.AddNode(&nodeOne);
    relationHandler.AddEdgeToNode(&nodeOne, &edgeOne);

    cout << edgeOne.GetN1()->GetId();
    return 0;
}