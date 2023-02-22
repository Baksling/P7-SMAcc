#include "domain_optimization_visitor.h"
#include <list>

void domain_optimization_visitor::visit(network* a)
{
    if(has_visited(a)) return;

    for (int i = 0; i < a->variables.size; ++i)
    {
        clock_var* var = &a->variables.store[i];
        variables_clock_map_.insert(std::pair<int, bool>(var->id, var->rate > 0));
    }
    
    accept(a, this);
}

void domain_optimization_visitor::visit(node* n)
{
    if(has_visited(n)) return;
    this->node_count++;
    this->node_map_.insert(std::pair<int,node*>(n->id, n));
    
    this->max_edge_fanout_ = std::max(this->max_edge_fanout_, static_cast<unsigned>(n->edges.size));
    for (int i = 0; i < n->invariants.size; ++i)
    {
        const constraint con = n->invariants.store[i];
        if(con.uses_variable)
        {
            this->contains_invalid_constraint_ = this->contains_invalid_constraint_ || expr_contains_clock(con.expression);
        }
    }
    
    accept(n, this);
}

void domain_optimization_visitor::visit(edge* e)
{
    if(has_visited(e)) return;
    
    compound_optimize_constraints(e);
    
    accept(e, this);
}

void domain_optimization_visitor::visit(constraint* c)
{
    if(has_visited(c)) return;
    
    accept(c, this);
}

void domain_optimization_visitor::visit(clock_var* cv)
{
    if(has_visited(cv)) return;

    accept(cv, this);
}

void domain_optimization_visitor::visit(update* u)
{
    if(has_visited(u)) return;
    
    accept(u, this);
}

void domain_optimization_visitor::visit(expr* ex)
{
    if(has_visited(ex)) return;

    bool has_lock = false;
    if(check_depth_lock_)
    {
        has_lock = true;
        check_depth_lock_ = false;
        const unsigned max_depth = count_expr_depth(ex);
        this->max_expr_depth_ = std::max(this->max_expr_depth_, max_depth);
    }
    
    accept(ex, this);

    if(has_lock) check_depth_lock_ = true;
}

void domain_optimization_visitor::clear()
{
    visitor::clear();
    this->node_count = 0;
    this->max_expr_depth_ = 0;
    this->max_edge_fanout_ = 0;
    this->check_depth_lock_ = true;
    this->variables_clock_map_.clear();
    this->check_depth_lock_ = true;
    this->node_map_.clear();
}

unsigned domain_optimization_visitor::get_max_expr_depth() const
{
    return max_expr_depth_;
}

bool domain_optimization_visitor::has_invalid_constraint() const
{
    return this->contains_invalid_constraint_;
}

unsigned domain_optimization_visitor::get_max_fanout() const
{
    return this->max_edge_fanout_;
}

unsigned domain_optimization_visitor::get_node_count() const
{
    return this->node_count;
}

std::unordered_map<int, node*> domain_optimization_visitor::get_node_map() const
{
    return std::unordered_map<int, node*>(this->node_map_);
}


unsigned domain_optimization_visitor::count_expr_depth(const expr* ex)
{
    const unsigned conditional = ex->operand == expr::conditional_ee ? count_expr_depth(ex->conditional_else) : 0;
    const unsigned left = ex->left != nullptr ? count_expr_depth(ex->left) : 0;
    const unsigned right = ex->right != nullptr ? count_expr_depth(ex->right) : 0;

    const unsigned temp = (left > right ? left : right);
    return (conditional > temp ? conditional : temp) + 1;
}

void domain_optimization_visitor::compound_optimize_constraints(edge* e)
{
    std::list<constraint> con_lst;
    //add all guards from edge
    for (int i = 0; i < e->guards.size; ++i)
    {
        con_lst.push_back(e->guards.store[i]);
    }

    //go through each invariant
    for (int i = 0; i < e->dest->invariants.size; ++i)
    {
        const constraint inv = e->dest->invariants.store[i];
        //if invariant does not use variable, 
        if(!inv.uses_variable)
        {
            con_lst.push_back(inv);
            continue;
        }
        bool any = false;
        for (int j = 0; j < e->updates.size; ++j)
        {
            const update upd = e->updates.store[j];
            if(upd.variable_id != inv.variable_id) continue;
            constraint compound = {};
            compound.operand = inv.operand;
            compound.value = upd.expression;
            compound.uses_variable = false;
            compound.expression = inv.expression;
            
            any = true;
            con_lst.push_back(compound);
        }
        if(!any) con_lst.push_back(inv);
    }

    if(con_lst.size() <= static_cast<size_t>(e->guards.size)) return;

    constraint* store = static_cast<constraint*>(malloc(sizeof(constraint)*con_lst.size()));
    int i = 0;
    for (const constraint& con : con_lst)
    {
        store[i++] = con;
    }
    e->guards = arr<constraint>{ store, i };
    
}

bool domain_optimization_visitor::expr_contains_clock(const expr* ex)
{
    if(ex->operand == expr::literal_ee) return false;
    if(ex->operand == expr::clock_variable_ee)
        return this->variables_clock_map_.at(ex->variable_id);

    const bool left = ex->left != nullptr ? expr_contains_clock(ex->left) : false;
    const bool right = ex->right != nullptr ? expr_contains_clock(ex->right) : false;
    const bool cond_else = ex->operand == expr::conditional_ee && ex->conditional_else != nullptr
                               ? expr_contains_clock(ex->conditional_else) : false;

    return left || right ||cond_else;
}


