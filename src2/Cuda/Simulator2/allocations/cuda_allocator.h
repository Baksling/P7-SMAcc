#pragma once
#include <unordered_map>

#include "memory_allocator.h"
#include "../Domain.h"

class cuda_allocator
{
    memory_allocator* allocator_;
    std::unordered_map<const node*, node*> circular_ref_{};
public:
    explicit cuda_allocator(memory_allocator* allocator)
    {
        this->allocator_ = allocator;
        this->circular_ref_ = std::unordered_map<const node*, node*>();
    }

    //The one to rule them all!
    automata* allocate_automata(const automata* source);
    
    void allocate_node(const node* source, node* dest);
    void allocate_edge(const edge* source, edge* dest);
    void allocate_constraint(const constraint* source, constraint* dest);
    void allocate_update(const update* source, update* dest);
    void allocate_clock(const clock_var* source, clock_var* dest) const;
    void allocate_expr(const expr* source, expr* dest);
};
