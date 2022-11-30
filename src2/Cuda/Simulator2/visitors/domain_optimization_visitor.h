#pragma once
#include "visitor.h"

class domain_optimization_visitor : public visitor
{
    unsigned max_expr_depth_ = 0;
public:
    void visit(automata* a) override;
    void visit(node* n) override;
    void visit(edge* e) override;
    void visit(constraint* c) override;
    void visit(clock_var* cv) override ;
    void visit(update* u) override;
    void visit(expr* ex) override;

    unsigned get_max_expr_depth() const;

    static unsigned count_expr_depth(const expr* ex);

};
