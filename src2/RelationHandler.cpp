using namespace std;

#include <iostream>
#include "Node.h"
#include "Edge.h"
#include "RelationHandler.h"

void RelationHandler::add_node(node* n)
{
    list<edge*> edgeList;
    nodeToEdgeList[n] = edgeList;
}

void RelationHandler::add_edge_to_node(node* n, edge* e)
{
    if(nodeToEdgeList.find(n) != nodeToEdgeList.end())
    {
        nodeToEdgeList[n].push_back(e);
    }
    else
    {
        cout << "I DO NOT CONTAIN THAT FUCKER!";
    }
}

list<edge*> RelationHandler::get_edges(node* n)
{
    return nodeToEdgeList[n];
}


