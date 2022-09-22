#ifndef NODE_H
#define NODE_H

#include <list>
#include "Edge.h"
#include "Guard.h"
#include "Timer.h"
#include "Update.h"

using namespace std;  // NOLINT(clang-diagnostic-header-hygiene)
class edge;
class timer;

class node {
private:
    list<edge> edges_;
    bool is_goal_;
    list<guard> invariants_;
    int id_;
public:
    node(int id, bool is_goal = false);
    int get_id();
    void add_edge(node* n, list<guard> guards, list<update>* updates);
    bool is_goal();
    void add_invariant(logical_operator type, double value, timer* timer);
    list<edge>* get_edges();
    bool validate_invariants();
    list<guard>* get_invariants();
        
};

#endif // NODE_H
