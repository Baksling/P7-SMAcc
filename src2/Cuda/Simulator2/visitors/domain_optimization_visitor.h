﻿#pragma once
#include <unordered_map>

#include "visitor.h"

class domain_optimization_visitor : public visitor
{
    unsigned max_expr_depth_ = 0;
    bool check_depth_lock = true;
    bool contains_invalid_constraint = false;
    std::unordered_map<int, bool> variables_clock_map_;
    
    struct model_size
    {
        size_t total_memory_size = 0;
        unsigned nodes = 0; 
        unsigned edges = 0; 
        unsigned constraint = 0;
        unsigned updates = 0;
        unsigned clocks = 0;
        unsigned expressions = 0;
    } model_counter_;
    
    static unsigned count_expr_depth(const expr* ex);
    static void compound_optimize_constraints(edge* e);
    bool expr_contains_clock(const expr* ex);
public:
    void optimize(automata* a){ visit(a);  }
    void visit(automata* a) override;
    void visit(node* n) override;
    void visit(edge* e) override;
    void visit(constraint* c) override;
    void visit(clock_var* cv) override ;
    void visit(update* u) override;
    void visit(expr* ex) override;

    void clear();
    
    unsigned get_max_expr_depth() const;
    bool invalid_constraint() const;
};
