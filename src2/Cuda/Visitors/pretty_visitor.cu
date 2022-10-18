#include "pretty_visitor.h"

#include "../Domain/constraint_t.h"
#include "../Domain/edge_t.h"
#include "../Domain/node_t.h"
#include "../Domain/stochastic_model_t.h"

void pretty_visitor::visit(constraint_t* constraint)
{
    if (constraint == nullptr) return;
    printf("Constraint type: %s | Timer 1 id: %3d | Timer 2 id: %3d | value: %10f \n", constraint_t::to_string(constraint->get_type()).c_str(),
           constraint->get_timer1_id(), constraint->get_timer2_id(), constraint->get_value());
    constraint->accept(this);
}

void pretty_visitor::visit(edge_t* edge)
{
    if (edge == nullptr) return;
    printf("    Edge id: %3d | Weight: %4f | Dest node: %3d \n", edge->get_id(), edge->get_weight(),
           edge->get_dest()->get_id());
    edge->accept(this);
}

void pretty_visitor::visit(node_t* node)
{
    if (node == nullptr) return;
    if (checker_.find(node) != checker_.end()) return;
    checker_.insert(node);
    printf("\nNode id: %3d | Is branch: %d | Is goal: %d \n", node->get_id(), node->is_branch_point(),
           node->is_goal_node());
    node->accept(this);
}

void pretty_visitor::visit(stochastic_model_t* model)
{
    if (model == nullptr) return;
    printf("Model start: \n");
    model->accept(this);
    printf("Model end\n");
    pretty_helper();
}

void pretty_visitor::visit(clock_variable* timer)
{
    if (timer == nullptr) return;
    //printf("|&|"); why we do this????
    printf("Timer id: %3d | Value: %10f \n", timer->get_id(), timer->get_time());
    timer->accept(this);
}

void pretty_visitor::visit(update_t* update)
{
    if (update == nullptr) return;
    // printf("        Update id: %3d | Timer id: %3d | Value: %10f \n", update->get_id(), update->get_timer_id(), update->get_timer_value());
    update->accept(this);
}

void pretty_visitor::visit(expression* expression)
{
    if (expression == nullptr) return;
    std::cout << expression->to_string();
    expression->accept(this);
}

void pretty_visitor::pretty_helper()
{
    printf("⠀⠀⠀⠀⠀⢰⡿⠋⠁⠀⠀⠈⠉⠙⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n");
    printf("⠀⠀⠀⠀⢀⣿⠇⠀⢀⣴⣶⡾⠿⠿⠿⢿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n");
    printf("⠀⠀⣀⣀⣸⡿⠀⠀⢸⣿⣇⠀⠀⠀⠀⠀⠀⠙⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n");
    printf("⠀⣾⡟⠛⣿⡇⠀⠀⢸⣿⣿⣷⣤⣤⣤⣤⣶⣶⣿⠇⠀⠀⠀⠀⠀⠀⠀⣀⠀⠀\n");
    printf("⢀⣿⠀⢀⣿⡇⠀⠀⠀⠻⢿⣿⣿⣿⣿⣿⠿⣿⡏⠀⠀⠀⠀⢴⣶⣶⣿⣿⣿⣆\n");
    printf("⢸⣿⠀⢸⣿⡇⠀⠀⠀⠀⠀⠈⠉⠁⠀⠀⠀⣿⡇⣀⣠⣴⣾⣮⣝⠿⠿⠿⣻⡟\n");
    printf("⢸⣿⠀⠘⣿⡇⠀⠀⠀⠀⠀⠀⠀⣠⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠉⠀\n");
    printf("⠸⣿⠀⠀⣿⡇⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠉⠀⠀⠀⠀\n");
    printf("⠀⠻⣷⣶⣿⣇⠀⠀⠀⢠⣼⣿⣿⣿⣿⣿⣿⣿⣛⣛⣻⠉⠁⠀⠀⠀⠀⠀⠀⠀\n");
    printf("⠀⠀⠀⠀⢸⣿⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀\n");
    printf("⠀⠀⠀⠀⢸⣿⣀⣀⣀⣼⡿⢿⣿⣿⣿⣿⣿⡿⣿⣿⡿\n");
    printf("         BIGUS DICKUS        \n");
}


