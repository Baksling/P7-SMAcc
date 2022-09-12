// Your First C++ Program

#include <iostream>
#include <string>
#include "Node.h"
#include "Edge.h"
using namespace std;

int main() {
    Node nodeOne(1);
    Node nodeTwo(2);

    Edge edgeOne(&nodeOne, &nodeTwo, 1);

    cout << edgeOne.GetN1()->outputs;
    return 0;
}