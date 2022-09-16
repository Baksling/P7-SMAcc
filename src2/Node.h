#ifndef NODE_H
#define NODE_H

#include <list>
#include "Edge.h"
#include "Guard.h"
#include "Timer.h"
#include "Update.h"
#include <cuda.h>
#include <cuda_runtime.h>
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
    __device__ int get_id();
    __host__ void add_edge(node* n, list<guard> guards, list<update>* updates);
    __host__ bool is_goal() const;
    __host__ void add_invariant(logical_operator type, double value, timer* timer);
    __host__ list<edge>* get_edges();
    __host__ bool validate_invariants();
    __host__ list<guard>* get_invariants();
        
};

#endif // NODE_H
