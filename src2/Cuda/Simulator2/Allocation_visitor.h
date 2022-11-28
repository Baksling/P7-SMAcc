#pragma once
#include <unordered_map>

#include "Domain.h"

class Allocation_visitor
{
private:
    std::unordered_map<void*, void*> mem_pointer_map_;
    void* get_mem(void* p) const;
    void cuda_allocate(void* p, const size_t size);
public:
    void visit(automata* a);
    void visit(node* n);
    void visit(edge* e);
    void visit(expr* e);
    void visit(clock_var* cv);
    void visit(constraint* c);
    void visit(update* u);
};
