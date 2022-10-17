#include "domain_analysis_visitor.h"

void domain_analysis_visitor::visit(constraint_t* constraint)
{
    return;
}

void domain_analysis_visitor::visit(edge_t* edge)
{
    if (edge == nullptr) return;
    edge->accept(this);
}

void domain_analysis_visitor::visit(node_t* node)
{
    if (node == nullptr) return;
    if (checker.find(node) != checker.end()) return;
    checker.insert(node);
    
    const lend_array<edge_t*> arr = node->get_edges();
    int acc = 0;
    for (int i = 0; i < arr.size(); ++i)
    {
        acc += arr.get(i)->get_updates_size();
    }
    if (acc > max_update_per_node_) max_update_per_node_ = acc;
    node->accept(this);
    
}

void domain_analysis_visitor::visit(stochastic_model_t* model)
{
    if (model == nullptr) return;
    model->accept(this);
}

void domain_analysis_visitor::visit(clock_timer_t* timer)
{
    return;
}

void domain_analysis_visitor::visit(update_t* update)
{
    if (update == nullptr) return;
    const int temp = update_t::get_expression_depth(update->get_expression_root());
    if(temp > max_expression_) max_expression_ = temp;
    update->accept(this);
}

void domain_analysis_visitor::visit(system_variable* variable)
{
    return;
}

void domain_analysis_visitor::visit(update_expression* expression)
{
    return;
}

std::tuple<int, int> domain_analysis_visitor::get_results()
{
    return {max_expression_, max_update_per_node_};
}

