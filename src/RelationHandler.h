#include <map>
#include <list>
#include "Node.h"
#include "Edge.h"

using namespace std;

#ifndef RELATIONHANDLER_H
#define RELATIONHANDLER_H

class RelationHandler
{
private:
    map<Node*, list<Edge*>> nodeToEdgeList;
public:
    void AddNode(Node* n);
    void AddEdgeToNode(Node* n, Edge* e);
    list<Edge> GetEdges(Node* n);
};


#endif // RELATIONHANDLER_H