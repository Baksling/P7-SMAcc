#include "pretty_visitor.h"

void pretty_visitor::visit(constraint_t* constraint)
{
    if (constraint == nullptr) return;
    printf("        Type: %d | Timer 1 id: %d | Timer 2 id: %d | value: %f \n", constraint->get_type(),
           constraint->get_timer1_id(), constraint->get_timer2_id(), constraint->get_value());
    constraint->accept(this);
        
}

void pretty_visitor::visit(edge_t* edge)
{
    if (edge == nullptr) return;

    printf("    Edge id: %d | Weight: %f | Dest node: %d \n", edge->get_id(), edge->get_weight(),
           edge->get_dest()->get_id());
    edge->accept(this);
}

void pretty_visitor::visit(node_t* node)
{
    if (node == nullptr) return;
    if (checker.find(node) == checker.end()) return;
    checker.insert(node);
    printf("Node id: %d | Is branch: %d | Is goal: %d \n", node->get_id(), node->is_branch_point(),
           node->is_goal_node());
    node->accept(this);
}

void pretty_visitor::visit(stochastic_model_t* model)
{
    if (model == nullptr) return;
    printf("Model start: \n");
    model->accept(this);
    printf("Model end:");
}

void pretty_visitor::visit(clock_timer_t* timer)
{
    if (timer == nullptr) return;
    printf("            Timer id: %d | Value: %f \n", timer->get_id(), timer->get_time());
    timer->accept(this);
}

void pretty_visitor::visit(update_t* update)
{
    if (update == nullptr) return;
    printf("        Update id: %d | Timer id: %d | Value: %f \n", update->get_id(), update->get_timer_id(), update->get_timer_value());
    update->accept(this);
        
}
