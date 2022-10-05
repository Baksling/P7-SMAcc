#include "cuda_visitor.h"

void cuda_visitor::visit(constraint_t* constraint)
{
    if (constraint == nullptr) return;
    constraint->accept(this);
}

void cuda_visitor::visit(edge_t* edge)
{
    if (edge == nullptr) return;
    edge->accept(this);
}

void cuda_visitor::visit(node_t* node)
{
    if (node == nullptr) return;
    node->accept(this);
}

void cuda_visitor::visit(stochastic_model_t* model)
{
    if (model == nullptr) return;
    model->accept(this);
}

void cuda_visitor::visit(clock_timer_t* timer)
{
    if (timer == nullptr) return;
    timer->accept(this);
}

void cuda_visitor::visit(update_t* update)
{
    if (update == nullptr) return;
    update->accept(this);    
}
