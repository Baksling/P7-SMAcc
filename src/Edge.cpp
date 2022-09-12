#include "Edge.h"
#include "Node.h"
using namespace std;

Edge::Edge(Node* _n1, Node* _n2, float _weight) {
    n1 = _n1;
    n2 = _n2;
    weight = _weight;
}

Node* Edge::GetN1() {
    return n1;
}

Node* Edge::GetN2() {
    return n2;
}

float Edge::GetWeight() {
    return weight;
}