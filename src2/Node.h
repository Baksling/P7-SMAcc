#ifndef NODE_H
#define NODE_H

#include <list>
#include "Edge.h"
using namespace std;  // NOLINT(clang-diagnostic-header-hygiene)
class edge;

class node {
private:
    list<edge*> edges_;
    bool is_goal_;
public:
    int id;
    int get_id() const;
    explicit node(int id, bool is_goal = false);
    void add_edge(edge* e);
    void add_edge(node* n, float weight);
    bool is_goal() const;
    list<edge*> get_edges();
        
};

#endif // NODE_H
