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
    map<node*, list<edge*>> nodeToEdgeList;
public:
    void add_node(node* n);
    void add_edge_to_node(node* n, edge* e);
    list<edge*> get_edges(node* n);
};


#endif // RELATIONHANDLER_H