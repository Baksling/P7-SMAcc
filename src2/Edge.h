#ifndef EDGE_H
#define EDGE_H

#include <list>
#include "Node.h"
#include "Guard.h"
#include "Timer.h"
class node;
class guard;
class timer;

using namespace std;

class edge {
private:
    node* node_;
    float weight_;
    list<guard*> guards_;
public:
    edge(node* n1, float weight);
    bool validate();
    node* get_node() const;
    float get_weight() const;
    void add_guard(guard* guard);
};



#endif // EDGE_H