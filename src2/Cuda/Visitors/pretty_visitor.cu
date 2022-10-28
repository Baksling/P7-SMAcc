#include "pretty_visitor.h"

#include "../Domain/expressions/constraint_t.h"
#include "../Domain/edge_t.h"
#include "../Domain/node_t.h"
#include "../Domain/stochastic_model_t.h"

void pretty_visitor::indentation() const
{
    for (int i = 0; i < scope_; ++i)
    {
        printf("    ");
    }

    //printf("-----scope: %d\n", this->scope_);
}

pretty_visitor::pretty_visitor()
{
    this->check_ = static_cast<int*>(malloc(3*sizeof(int)));
    for (int i = 0; i < 3; ++i)
    {
        check_[i] = 0;
    }
}

void pretty_visitor::visit(constraint_t* constraint)
{
    if (constraint == nullptr)
    {
        return;
    }
    indentation();
    constraint->pretty_print();
    scope_++;
    constraint->accept(this);
    scope_--;
}

void pretty_visitor::visit(edge_t* edge)
{
    if (edge == nullptr)
    {
        return;
    }
    indentation();
    edge->pretty_print();
    scope_++;
    edge->accept(this);
    scope_--;
}

void pretty_visitor::visit(node_t* node)
{
    if (node == nullptr)
    {
        return;
    }
    const bool is_new = checker_.insert(node->get_id()).second;
    if(!is_new) return;

    //std::cout << "set count for node :" << checker_.count(node) << "\n";
    // if (checker_.count(node))
    // {
    //     return;
    // }
    //if(check_[node->get_id()] > 0) return;
    //if(checker_.count(node) == 1) return;
    //check_[node->get_id()] = 1;
    //std::cout << "set count for node after insert:" << checker_.count(node) << "\n";
    checker_.insert(node->get_id());
    this->scope_ = 0;
    indentation();
    node->pretty_print();
    scope_++;
    node->accept(this);
    scope_--;

}

void pretty_visitor::visit(stochastic_model_t* model)
{
    if (model == nullptr) return;
    model->pretty_print();
    model->accept(this);
    printf("Model end\n");
    pretty_helper();
}

void pretty_visitor::visit(clock_variable* timer)
{
    if (timer == nullptr)
    {
        return;
    }
    indentation();
    timer->pretty_print();
    scope_++;
    timer->accept(this);
    scope_--;
}

void pretty_visitor::visit(update_t* update)
{
    if (update == nullptr)
    {
        return;
    }
    indentation();
    update->pretty_print();
    scope_++;
    update->accept(this);
    scope_--;
}

void pretty_visitor::visit(expression* expression)
{
    if (expression == nullptr)
    {
        return;
    }
    //indentation();
    //std::cout << expression->to_string();
    scope_++;
    expression->accept(this);
    scope_--;
}

void pretty_visitor::pretty_helper()
{
    // printf("⠀⠀⠀⠀⠀⢰⡿⠋⠁⠀⠀⠈⠉⠙⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n");
    // printf("⠀⠀⠀⠀⢀⣿⠇⠀⢀⣴⣶⡾⠿⠿⠿⢿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n");
    // printf("⠀⠀⣀⣀⣸⡿⠀⠀⢸⣿⣇⠀⠀⠀⠀⠀⠀⠙⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n");
    // printf("⠀⣾⡟⠛⣿⡇⠀⠀⢸⣿⣿⣷⣤⣤⣤⣤⣶⣶⣿⠇⠀⠀⠀⠀⠀⠀⠀⣀⠀⠀\n");
    // printf("⢀⣿⠀⢀⣿⡇⠀⠀⠀⠻⢿⣿⣿⣿⣿⣿⠿⣿⡏⠀⠀⠀⠀⢴⣶⣶⣿⣿⣿⣆\n");
    // printf("⢸⣿⠀⢸⣿⡇⠀⠀⠀⠀⠀⠈⠉⠁⠀⠀⠀⣿⡇⣀⣠⣴⣾⣮⣝⠿⠿⠿⣻⡟\n");
    // printf("⢸⣿⠀⠘⣿⡇⠀⠀⠀⠀⠀⠀⠀⣠⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠉⠀\n");
    // printf("⠸⣿⠀⠀⣿⡇⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠉⠀⠀⠀⠀\n");
    // printf("⠀⠻⣷⣶⣿⣇⠀⠀⠀⢠⣼⣿⣿⣿⣿⣿⣿⣿⣛⣛⣻⠉⠁⠀⠀⠀⠀⠀⠀⠀\n");
    // printf("⠀⠀⠀⠀⢸⣿⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀\n");
    // printf("⠀⠀⠀⠀⢸⣿⣀⣀⣀⣼⡿⢿⣿⣿⣿⣿⣿⡿⣿⣿⡿\n");
    // printf("         BIGUS DICKUS        \n");


    printf("⠀⠀⠀⠀⠀⢀⣴⡾⠿⠿⠿⠿⢶⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n");
    printf("⠀⠀⠀⠀⢠⣿⠁⠀⠀⠀⣀⣀⣀⣈⣻⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀\n");
    printf("⠀⠀⠀⠀⣾⡇⠀⠀⣾⣟⠛⠋⠉⠉⠙⠛⢷⣄⠀⠀⠀⠀⠀⠀⠀\n");
    printf("⢀⣤⣴⣶⣿⠀⠀⢸⣿⣿⣧⠀⠀⠀⠀⢀⣀⢹⡆⠀⠀⠀⠀⠀⠀\n");
    printf("⢸⡏⠀⢸⣿⠀⠀⠀⢿⣿⣿⣷⣶⣶⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀\n");
    printf("⣼⡇⠀⢸⣿⠀⠀⠀⠈⠻⠿⣿⣿⠿⠿⠛⢻⡇⠀⠀⠀⠀⠀⠀⠀\n");
    printf("⣿⡇⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣼⣷⣶⣶⣶⣤⡀⠀⠀\n");
    printf("⣿⡇⠀⢸⣿⠀⠀⠀⠀⠀⠀⣀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀\n");
    printf("⢻⡇⠀⢸⣿⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⡿⠿⣿⣿⣿⣿⡇\n");
    printf("⠈⠻⠷⠾⣿⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⢸⣿⣿⣿⣇\n");
    printf("⠀⠀⠀⠀⣿⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⢸⣿⣿⣿⡿\n");
    printf("⠀⠀⠀⠀⢿⣧⣀⣠⣴⡿⠙⠛⠿⠿⠿⠿⠉⠀⠀⢠⣿⣿⣿⣿⠇\n");
    printf("⠀⠀⠀⠀⠀⢈⣩⣭⣥⣤⣤⣤⣤⣤⣤⣤⣤⣤⣶⣿⣿⣿⣿⠏⠀\n");
    printf("⠀⠀⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠀⠀\n");
    printf("⠀⠀⠀⢸⣿⣿⣿⡟⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠋⠁⠀⠀⠀⠀\n");
    printf("⠀⠀⠀⢸⣿⣿⣿⣷⣄⣀⣀⣀⣀⣀⣀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀\n");
    printf("⠀⠀⠀⠀⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀\n");
    printf("⠀⠀⠀⠀⠀⠈⠛⠿⠿⣿⣿⣿⣿⣿⠿⠿⢿⣿⣿⣿⣿⣿⡄⠀⠀\n");
    printf("⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⢀⣹⣿⣿⣿⡇⠀⠀\n");
    printf("⠀⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠀⠀\n");
    printf("⠀⠀⠀⠀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠁⠀⠀⠀\n");
    printf("⠀⠀⠀⠀⣿⣿⣿⣿⠁⠀⠀⠀⠀⠀⠉⠉⠁⢤⣤⣤⣤⣤⣤⣤⡀\n");
    printf("⠀⠀⠀⠀⢿⣿⣿⣿⣷⣶⣶⣶⣶⣾⣿⣿⣿⣆⢻⣿⣿⣿⣿⣿⡇\n");
    printf("⠀⠀⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠻⣿⣿⣿⡿⠁\n");
    printf("⠀⠀⠀⠀⠀⠀⠈⠙⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠉⠀⠙⠛⠉⠀⠀\n");
    printf("          BIGUS DICKUS        \n");
}



