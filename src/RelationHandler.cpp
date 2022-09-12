using namespace std;

#include "Node.h"
#include "Edge.h"
#include "RelationHandler.h"

#define BUFFER_SIZE 10

void RelationHandler::AddNode(Node* n)
{
    list<Edge*> edgeList;
    nodeToEdgeList[n] = edgeList;
}

void RelationHandler::AddEdgeToNode(Node* n, Edge* e)
{
    nodeToEdgeList[n].push_back(e);
}