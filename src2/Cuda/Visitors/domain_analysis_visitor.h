#pragma once

#ifndef DOMAIN_ANALYSIS_VISITOR_H
#define DOMAIN_ANALYSIS_VISITOR_H

#include "visitor.h"
#include <set>

class domain_analysis_visitor : public visitor
{
private:
    std::set<node_t*> checker = {};
    int max_expression_ = 0;
    int max_update_per_node_ = 0;
    
public:
    void visit(constraint_t* constraint) override;
    void visit(edge_t* edge) override;
    void visit(node_t* node) override;
    void visit(stochastic_model_t* model) override;
    void visit(clock_variable* timer) override;
    void visit(update_t* update) override;
    void visit(expression* expression) override;
    std::tuple<int, int> get_results();
};

#endif
