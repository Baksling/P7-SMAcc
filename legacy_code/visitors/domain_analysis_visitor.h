#pragma once

#ifndef DOMAIN_ANALYSIS_VISITOR_H
#define DOMAIN_ANALYSIS_VISITOR_H

#include "visitor.h"
#include <set>

class domain_analysis_visitor : public visitor
{
private:
    std::set<node_t*> checker = {};
    unsigned int max_expression_ = 0;
    unsigned int max_update_per_node_ = 0;
    
public:
    void visit(constraint_t* constraint) override;
    void visit(edge_t* edge) override;
    void visit(node_t* node) override;
    void visit(stochastic_model_t* model) override;
    void visit(clock_variable* timer) override;
    void visit(update_t* update) override;
    void visit(expression* expression) override;
    unsigned get_max_expression_depth() const;
    unsigned get_max_update_width() const;
};

#endif
