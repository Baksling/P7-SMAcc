#pragma once
#include <unordered_map>

#include "visitor.h"
#include "../engine/model_oracle.h"



class domain_optimization_visitor : public visitor
{
    std::unordered_set<std::string>* query_;
    std::unordered_map<int, int>* node_subsystems_map_;
    std::unordered_map<int, std::string>* node_names_;
    std::unordered_map<int, std::string>* subsystem_names_;
    
    unsigned node_count_ = 0;
    unsigned max_expr_depth_ = 0;
    unsigned max_edge_fanout_ = 0;
    bool check_depth_lock_ = true;
    bool contains_invalid_constraint_ = false;
    std::unordered_map<int, bool> variables_clock_map_;
    std::unordered_map<int,node*> node_map_;

    bool is_goal(const int node_id) const;
    static unsigned count_expr_depth(const expr* ex);
    static void compound_optimize_constraints(edge* e);
    bool expr_contains_clock(const expr* ex);
public:
    explicit domain_optimization_visitor(
        std::unordered_set<std::string>* query,
        std::unordered_map<int, int>* node_subsystems_map,
        std::unordered_map<int, std::string>* node_names,
        std::unordered_map<int, std::string>* subsystem_names)
    {
        this->query_ = query;
        this->node_subsystems_map_ = node_subsystems_map;
        this->node_names_ = node_names;
        this->subsystem_names_ = subsystem_names;
    }
    void optimize(network* a){ visit(a);  }
    void visit(network* a) override;
    void visit(node* n) override;
    void visit(edge* e) override;
    void visit(constraint* c) override;
    void visit(clock_var* cv) override ;
    void visit(update* u) override;
    void visit(expr* ex) override;

    void clear() override;
    
    unsigned get_max_expr_depth() const;
    bool has_invalid_constraint() const;
    unsigned get_max_fanout() const;
    unsigned get_node_count() const;
    std::unordered_map<int, node*> get_node_map() const;
};
