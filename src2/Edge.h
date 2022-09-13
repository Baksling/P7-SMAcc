#ifndef EDGE_H
#define EDGE_H

#include "Node.h"
class node;

class edge {
private:
    node* node_;
    float weight_;
public:
    edge(node* n1, float weight);
    bool validate();
    node* get_node() const;
    float get_weight() const;
};



#endif // EDGE_H