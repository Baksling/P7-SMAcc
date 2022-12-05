#include "memory_alignment_visitor.h"

void memory_alignment_visitor::post_process() const
{
    edge* e_point = this->oracle_.edge_point();
    for (unsigned i = 0; i < this->oracle_.model_counter.edges; ++i)
    {
        edge* e = &e_point[i];
        e->dest = static_cast<node*>(this->location_mapper_.at(e->dest));
    }
}

model_oracle memory_alignment_visitor::align(network* n, const model_size& model_m, memory_allocator* allocator)
{
    this->location_mapper_.clear();
    void* memory;
    allocator->allocate_host(&memory, model_m.total_memory_size());
    this->oracle_ = model_oracle(memory, model_m );
    this->move_state_ = model_size{};

    visit(n);

    post_process();
    
    return this->oracle_;
}

void memory_alignment_visitor::visit(network* nn)
{
    if(has_visited(nn)) return;
    this->move_state_.network_size = nn->automatas.size;

    network* net = this->oracle_.network_point();
    *net = *nn;
    this->location_mapper_.insert(std::pair<void*, void*>(nn, net));

    net->automatas.store = this->oracle_.network_nodes_point();
    net->variables.store = this->oracle_.variable_point();
    accept(nn, this);

    for (int i = 0; i < net->automatas.size; ++i)
    {
        net->automatas.store[i] = static_cast<node*>(this->location_mapper_.at(nn->automatas.store[i]));
    }
}

void memory_alignment_visitor::visit(node* n)
{
    if(has_visited(n)) return;
    
    node* no = &this->oracle_.node_point()[this->move_state_.nodes++];
    *no = *n;
    this->location_mapper_.insert(std::pair<void*, void*>(n, no));

    //this assumes invariants are visited first, and then edges, and then destination nodes.
    no->invariants.store  = &this->oracle_.constraint_point()[this->move_state_.constraints];
    no->edges.store = &this->oracle_.edge_point()[this->move_state_.edges];
    accept(n, this);

    no->lamda = static_cast<expr*>(this->location_mapper_.at(n->lamda));
}

void memory_alignment_visitor::visit(edge* e)
{
    if(has_visited(e)) return;

    edge* ed = &this->oracle_.edge_point()[this->move_state_.edges++];
    *ed = *e;
    this->location_mapper_.insert(std::pair<void*, void*>(e, ed));

    ed->guards.store = &this->oracle_.constraint_point()[this->move_state_.constraints];
    ed->updates.store = &this->oracle_.update_point()[this->move_state_.updates];
    
    accept(e, this);

    ed->weight = static_cast<expr*>(this->location_mapper_.at(e->weight));
}

void memory_alignment_visitor::visit(constraint* c)
{
    if(has_visited(c)) return;

    constraint* co = &this->oracle_.constraint_point()[this->move_state_.constraints++];
    *co = *c;
    this->location_mapper_.insert(std::pair<void*, void*>(c, co));

    accept(c, this);

    if(!c->uses_variable)
        co->value = static_cast<expr*>(this->location_mapper_.at(c->value));
    co->expression = static_cast<expr*>(this->location_mapper_.at(c->expression));
}

void memory_alignment_visitor::visit(clock_var* cv)
{
    if(has_visited(cv)) return;
    
    clock_var* cov = &this->oracle_.variable_point()[this->move_state_.variables++];

    *cov = *cv;
    this->location_mapper_.insert(std::pair<void*, void*>(cv, cov));
    
    accept(cv, this);
}

void memory_alignment_visitor::visit(update* u)
{
    if(has_visited(u)) return;

    update* up = &this->oracle_.update_point()[this->move_state_.updates++];
    *up = *u;
    this->location_mapper_.insert(std::pair<void*, void*>(u, up)); 
    
    accept(u, this);

    up->expression = static_cast<expr*>(this->location_mapper_.at(u->expression));
}

void memory_alignment_visitor::visit(expr* ex)
{
    if(has_visited(ex)) return;
    
    expr* exp = &this->oracle_.expression_point()[this->move_state_.expressions++];
    *exp = *ex;
    this->location_mapper_.insert(std::pair<void*, void*>(ex, exp));

    accept(ex, this);

    if(ex->left  != nullptr) exp->left  = static_cast<expr*>(this->location_mapper_.at(ex->left)); 
    if(ex->right != nullptr) exp->right = static_cast<expr*>(this->location_mapper_.at(ex->right));

    if(ex->operand == expr::conditional_ee && ex->conditional_else != nullptr)
        exp->conditional_else = static_cast<expr*>(this->location_mapper_.at(ex->conditional_else));
}
