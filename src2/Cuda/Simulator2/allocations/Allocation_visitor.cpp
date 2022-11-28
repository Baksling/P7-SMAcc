#include "Allocation_visitor.h"
void accept(automata* a, Allocation_visitor* v)
{
    if (a->network.store != nullptr)
    {
        for (int i = 0; i < a->network.size; ++i)
        {
            v->visit(a->network.store[i]);
        }
    }

    if (a->variables.store != nullptr)
    {
        for (int i = 0; i < a->variables.size; ++i)
        {
            v->visit(&a->variables.store[i]);
        }
    }
    
}

void accept(node* n, Allocation_visitor* v)
{
    // int id;
    // expr* lamda;
    // arr<edge> edges;
    // arr<constraint> invariants;
    // bool is_branch_point;
    // bool is_goal;

    if(n->lamda != nullptr)
        v->visit(n->lamda);

    if (n->invariants.store != nullptr)
    {
        for (int i = 0; i < n->invariants.size; ++i)
        {
            v->visit(&n->invariants.store[i]);
        }
    }
        
    if (n->edges.store != nullptr)
    {
        for (int i = 0; i < n->edges.size; ++i)
        {
            v->visit(&n->edges.store[i]);
        }
    }
    
}
void accept(edge* e, Allocation_visitor* v)
{
    // int channel;
    // expr* weight;
    // node* dest;
    // arr<constraint> guards;
    // arr<update> updates;

    if (e->weight != nullptr)
        v->visit(e->weight);
    if (e->dest != nullptr)
        v->visit(e->dest);

    if (e->guards.store != nullptr)
    {
        for (int i = 0; i < e->guards.size; ++i)
        {
            v->visit(&e->guards.store[i]);
        }
    }
        
    if (e->updates.store != nullptr)
    {
        for (int i = 0; i < e->updates.size; ++i)
        {
            v->visit(&e->updates.store[i]);
        }
    }
}
void accept(expr* e, Allocation_visitor* v)
{
    if (e->left != nullptr)
        v->visit(e->left);
    if (e->right != nullptr)
        v->visit(e->right);
    if(e->operand == expr::conditional_ee && e->conditional_else != nullptr )
        v->visit(e->conditional_else);
}

void accept(constraint* c, Allocation_visitor* v)
{
    if (!c->uses_variable && c->value != nullptr)
        v->visit(c->value);
    if (c->expression != nullptr) 
        v->visit(c->expression);
}
void accept(update* u, Allocation_visitor* v)
{
    if (u->expression != nullptr)
        v->visit(u->expression);
}

void* Allocation_visitor::get_mem(void* p) const
{
    if(mem_pointer_map_.count(p) == 0)
        return nullptr;
    else
        return mem_pointer_map_.at(p);
}

void Allocation_visitor::cuda_allocate(void* p, const size_t size)
{
    void* cuda_p = nullptr;
    cudaMalloc(&cuda_p, size);
    cudaMemcpy(cuda_p, p,size, cudaMemcpyHostToDevice);
    mem_pointer_map_.insert(p, cuda_p);
}

void Allocation_visitor::visit(automata* a)
{
    accept(a, this);

    automata temp;
    temp.variables = a->variables;
    temp.network = 
}

void Allocation_visitor::visit(clock_var* cv)
{
    if (mem_pointer_map_.count(cv) > 0)
        return;
    
    cuda_allocate(cv, sizeof(clock_var)); //maybe
}

void Allocation_visitor::visit(node* n)
{
    if (mem_pointer_map_.count(n) > 0)
        return;

    accept(n, this);

    node temp{};
    temp.id = n->id;
    temp.is_branch_point = n->is_branch_point;
    temp.is_goal = n->is_goal;
    temp.invariants = arr<constraint>{static_cast<constraint*>(malloc(n->invariants.size * sizeof(constraint))), n->invariants.size};
    temp.edges = arr<edge>{static_cast<edge*>(malloc(n->edges.size * sizeof(edge))), n->edges.size};

    for (int i = 0; i < n->invariants.size; ++i)
    {
        temp.invariants.store[i] = *static_cast<constraint*>(get_mem(&n->invariants.store[i]));
    }
    
    for (int i = 0; i < n->edges.size; ++i)
    {
        temp.edges.store[i] = *static_cast<edge*>(get_mem(&n->edges.store[i]));
    }
    
    constraint* invariant_p = nullptr;
    cudaMalloc(&invariant_p, n->invariants.size*sizeof(constraint));
    cudaMemcpy(invariant_p, n->invariants.store,n->invariants.size*sizeof(constraint), cudaMemcpyHostToDevice);

    edge* edge_p = nullptr;
    cudaMalloc(&edge_p, n->edges.size*sizeof(edge));
    cudaMemcpy(edge_p, n->edges.store,n->edges.size*sizeof(edge), cudaMemcpyHostToDevice);

    free(temp.invariants.store);
    free(temp.edges.store);
    
    temp.invariants = arr<constraint>{invariant_p, n->invariants.size};
    temp.edges = arr<edge>{edge_p, n->edges.size};

    cuda_allocate(&temp, sizeof(node));
}

void Allocation_visitor::visit(edge* e)
{
    if (mem_pointer_map_.count(e) > 0)
        return;

    accept(e, this);

    edge* temp = new edge;
    temp->dest = static_cast<node*>(get_mem(e->dest));
    temp->weight = static_cast<expr*>(get_mem(e->weight));
    temp->channel = e->channel;
    temp->guards = arr<constraint>{static_cast<constraint*>(malloc(e->guards.size * sizeof(constraint))), e->guards.size};
    temp->updates = arr<update>{static_cast<update*>(malloc(e->updates.size * sizeof(update))), e->updates.size};
    
    for (int i = 0; i < e->guards.size; ++i)
    {
        temp->guards.store[i] = *static_cast<constraint*>(get_mem(&e->guards.store[i]));
    }
    
    for (int i = 0; i < e->updates.size; ++i)
    {
        temp->updates.store[i] = *static_cast<update*>(get_mem(&e->updates.store[i]));
    }
    constraint* constraint_p = nullptr;
    cudaMalloc(&constraint_p, e->guards.size*sizeof(constraint));
    cudaMemcpy(constraint_p, e->guards.store,e->guards.size*sizeof(constraint), cudaMemcpyHostToDevice);

    update* update_p = nullptr;
    cudaMalloc(&update_p, e->updates.size*sizeof(update));
    cudaMemcpy(update_p, e->updates.store,e->updates.size*sizeof(update), cudaMemcpyHostToDevice);

    free(temp->guards.store);
    free(temp->updates.store);
    
    temp->guards = arr<constraint>{constraint_p, e->guards.size};
    temp->updates = arr<update>{update_p, e->updates.size};
    
    mem_pointer_map_.insert(e, temp);
}

void Allocation_visitor::visit(update* u)
{
    if (mem_pointer_map_.count(u) > 0)
        return;

    accept(u,this);

    update* temp = new update;
    temp->expression = static_cast<expr*>(get_mem(u->expression));
    temp->variable_id = u->variable_id;

    mem_pointer_map_.insert(u, temp);
}

void Allocation_visitor::visit(constraint* c)
{
    if (mem_pointer_map_.count(c) > 0)
        return;
    
    accept(c, this);
    
    constraint* temp = new constraint;
    temp->operand = c->operand;
    temp->expression = static_cast<expr*>(get_mem(c->expression));

    if(!c->uses_variable)
        temp->value = static_cast<expr*>(get_mem(c->value));
    else
        temp->variable_id = c->variable_id;

    mem_pointer_map_.insert(c, temp);
}

void Allocation_visitor::visit(expr* e)
{
    if (mem_pointer_map_.count(e) > 0)
        return;
    
    if (IS_LEAF(e->operand))
    {
        expr* cuda_p = nullptr;
        cudaMalloc(&cuda_p, sizeof(expr));
        cudaMemcpy(cuda_p, e,sizeof(expr), cudaMemcpyHostToDevice);
        mem_pointer_map_.insert(e, cuda_p);
        return;
    }
    accept(e, this);
    
    expr temp{};
    temp.left = static_cast<expr*>(get_mem(e->left));
    temp.right = static_cast<expr*>(get_mem(e->right));
    temp.conditional_else = static_cast<expr*>(get_mem(e->conditional_else));
    temp.operand = e->operand;
    
    cuda_allocate(&temp, sizeof(expr));
}
