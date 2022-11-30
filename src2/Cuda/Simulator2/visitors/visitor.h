#pragma once

#include <unordered_set>

#include "../Domain.h"

class visitor
{
private:
    std::unordered_set<const void*> visited_;

    bool check_visit(const void* p);

public:
    virtual ~visitor() = default;
    virtual void visit(automata* a) = 0;
    virtual void visit(node* n) = 0;
    virtual void visit(edge* e) = 0;
    virtual void visit(constraint* c) = 0;
    virtual void visit(clock_var* cv) = 0 ;
    virtual void visit(update* u) = 0;
    virtual void visit(expr* u) = 0;
    
    static void accept(const automata* a, visitor* v);
    static void accept(const node* n, visitor* v);
    static void accept(const edge* e, visitor* v);
    static void accept(const constraint* c, visitor* v);
    static void accept(const clock_var* c, visitor* v);
    static void accept(const update* u, visitor* v);
    static void accept(const expr* ex, visitor* v);

};
