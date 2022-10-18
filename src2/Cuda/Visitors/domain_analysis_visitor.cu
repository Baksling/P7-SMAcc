#include "domain_analysis_visitor.h"
#include "../Domain/constraint_t.h"
#include "../Domain/edge_t.h"
#include "../Domain/node_t.h"
#include "../Domain/stochastic_model_t.h"
#include "../common/lend_array.h"


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

void domain_analysis_visitor::visit(clock_variable* timer)
{
    return;
}

void domain_analysis_visitor::visit(update_t* update)
{
    if (update == nullptr) return;
    const unsigned int temp = update->get_expression_depth();
    if(temp > max_expression_) max_expression_ = temp;
    update->accept(this);
}


void domain_analysis_visitor::visit(expression* expression)
{
    return;
}

std::tuple<int, int> domain_analysis_visitor::get_results()
{
    return {max_expression_, max_update_per_node_};
}

