#include "Node.h"

#ifndef EDGE_H
#define EDGE_H

class Edge {
    private:
        Node* n1;
        Node* n2;
        float weight;
    public:
        Edge(Node* n1, Node* n2, float weight);
        float GetWeight();
        Node* GetN1();
        Node* GetN2();
};



#endif // EDGE_H