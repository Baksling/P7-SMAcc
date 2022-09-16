#ifndef EDGE_H
#define EDGE_H

#include <list>
#include "Node.h"
#include "Guard.h"
#include "Timer.h"
#include "Update.h"

class node;
class guard;
class timer;
class update;

using namespace std;

class edge {
private:
    node* node_;
    float weight_;
    list<guard> guards_;
    list<update>* updates_;
public:
    edge(node* n1, const list<guard>& guards, list<update>* updates);
    edge();
    bool validate();
    node* get_node() const;
    float get_weight() const;
    void add_guard(guard guard);
    void activate();
};



#endif // EDGE_H