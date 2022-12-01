#include "visitor.h"

bool visitor::has_visited(const void* p)
{
    if(visit_set_.count(p)) return true;
    visit_set_.insert(p);
    return false;
}

void visitor::accept(const automata* a, visitor* v)
{
    v->visit_set_.clear();
    for (int i = 0; i < a->network.size; ++i)
    {
        v->visit(a->network.store[i]);
    }

    for (int i = 0; i < a->variables.size; ++i)
    {
        v->visit(&a->variables.store[i]);
    }
}

void visitor::accept(const node* n, visitor* v)
{
    // if(v->has_visited(n)) return;
    
    for (int i = 0; i < n->edges.size; ++i)
    {
        v->visit(&n->edges.store[i]);
    }
    
    for (int i = 0; i < n->invariants.size; ++i)
    {
        v->visit(&n->invariants.store[i]);
    }

    for (int i = 0; i < n->edges.size; ++i)
    {
        v->visit(n->edges.store[i].dest);
    }
    
    v->visit(n->lamda);
}

void visitor::accept(const edge* e, visitor* v)
{
    // if(v->has_visited(e)) return;

    //This is handled by nodes accept method.
    // v->visit(e->dest);
    
    for (int i = 0; i < e->guards.size; ++i)
    {
        v->visit(&e->guards.store[i]);
    }

    for (int i = 0; i < e->updates.size; ++i)
    {
        v->visit(&e->updates.store[i]);
    }

    v->visit(e->weight);
}

void visitor::accept(const constraint* c, visitor* v)
{
    // if(v->has_visited(c)) return;

    if(!c->uses_variable)
    {
        v->visit(c->value);
    }
    v->visit(c->expression);
}

void visitor::accept(const clock_var* c, visitor* v)
{
    // if(v->has_visited(c)) return;
    return;
}

void visitor::accept(const update* u, visitor* v)
{
    // if(v->has_visited(u)) return;
    v->visit(u->expression);
}

void visitor::accept(const expr* ex, visitor* v)
{
    // if(v->has_visited(ex)) return;

    if(ex->left != nullptr)
    {
        v->visit(ex->left);
    }

    if(ex->right != nullptr)
    {
        v->visit(ex->right);
    }

    if(ex->operand == expr::conditional_ee && ex->conditional_else != nullptr)
    {
        v->visit(ex->conditional_else);
    }
}








