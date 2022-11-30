#include "domain_optimization_visitor.h"

void domain_optimization_visitor::visit(automata* a)
{
    accept(a, this);
}

void domain_optimization_visitor::visit(node* n)
{
    accept(n, this);
}

void domain_optimization_visitor::visit(edge* e)
{
    accept(e, this);
}

void domain_optimization_visitor::visit(constraint* c)
{
    accept(c, this);
}

void domain_optimization_visitor::visit(clock_var* cv)
{
    accept(cv, this);
}

void domain_optimization_visitor::visit(update* u)
{
    accept(u, this);
}

void domain_optimization_visitor::visit(expr* ex)
{
    const unsigned max_depth = count_expr_depth(ex);
    this->max_expr_depth_ = max_depth > this->max_expr_depth_ ? max_depth : this->max_expr_depth_; 
}

unsigned domain_optimization_visitor::get_max_expr_depth() const
{
    return max_expr_depth_;
}

unsigned domain_optimization_visitor::count_expr_depth(const expr* ex)
{
    const unsigned conditional = ex->operand == expr::conditional_ee ? count_expr_depth(ex->conditional_else) : 0;
    const unsigned left = ex->left != nullptr ? count_expr_depth(ex->left) : 0;
    const unsigned right = ex->right != nullptr ? count_expr_depth(ex->right) : 0;

    const unsigned temp = (left > right ? left : right);
    return (conditional > temp ? conditional : temp) + 1;
}
