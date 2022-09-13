#ifndef NODE_H
#define NODE_H

#include <list>
#include "Edge.h"
#include "Guard.h"
#include "Timer.h"
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
    int get_id() const;
    explicit node(int id, bool is_goal = false);
    void add_edge(node* n, float weight);
    bool is_goal() const;
    void add_guard(logical_operator type, double value, timer* timer);
    list<edge> get_edges();
    bool validate_invariants();
        
};

#endif // NODE_H
