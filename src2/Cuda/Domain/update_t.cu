#include "update_t.h"

update_t::update_t(const update_t* source, expression* expression)
{
    this->id_ = source->id_;
    this->variable_id_ = source->variable_id_;
    this->is_clock_update_ = source->is_clock_update_;
    this->expression_ = expression;
}

update_t::update_t(const int id, const int variable_id, const bool is_clock_update, expression* expression)
{
    this->id_ = id;
    this->variable_id_ = variable_id;
    this->is_clock_update_ = is_clock_update;
    this->expression_ = expression;
}

CPU GPU void update_t::apply_update(simulator_state* state) const
{
    const double value = state->evaluate_expression(this->expression_);
    if(this->is_clock_update_)
    {
        state->timers.at(this->variable_id_)->set_time(value);
    }
    else
    {
        //value is rounded correctly by adding 0.5. casting always rounds down.
        state->variables.at(this->variable_id_)->set_time(static_cast<int>(value + 0.5));  // NOLINT(bugprone-incorrect-roundings)
    }
}

CPU GPU void update_t::apply_temp_update(simulator_state* state) const
{
    const double value = state->evaluate_expression(this->expression_);
    if(this->is_clock_update_)
    {
        state->timers.at(this->variable_id_)->set_temp_time(value);
    }
    else
    {
        //value is rounded correctly by adding 0.5. casting always rounds down.
        state->variables.at(this->variable_id_)
            ->set_temp_time(static_cast<int>(value + 0.5));  // NOLINT(bugprone-incorrect-roundings)
    }
}

void update_t::reset_temp_update(const simulator_state* state) const
{
    if(this->is_clock_update_)
    {
        state->timers.at(this->variable_id_)->reset_temp_time();
    }
    else
    {
        //value is rounded correctly by adding 0.5. casting always rounds down.
        state->variables.at(this->variable_id_)->reset_temp_time();  // NOLINT(bugprone-incorrect-roundings)
    }
}

void update_t::accept(visitor* v) const
{
    v->visit(this->expression_);
}

CPU GPU int update_t::get_timer_id() const
{
    return this->variable_id_;
}

int update_t::get_id() const
{
    return this->id_;
}

void update_t::cuda_allocate(update_t* cuda, const allocation_helper* helper) const
{
    expression* expr = nullptr;
    cudaMalloc(&expr, sizeof(expression));
    helper->free_list->push_back(expr);
    this->expression_->cuda_allocate(expr, helper);
    
    const update_t copy = update_t(this, expr);
    cudaMemcpy(cuda, &copy, sizeof(update_t), cudaMemcpyHostToDevice);
}

update_expression* update_t::get_expression_root() const
{
    return this->expression_;
}

int update_t::get_expression_depth(const update_expression* exp)
{
    int left, right;
    if(exp->get_left() == nullptr) left = 0;
    else left = get_expression_depth(exp->get_left());
    if(exp->get_right() == nullptr) right = 0;
    else right = get_expression_depth(exp->get_right());

    return ((left<right)?right:left) + 1;
}
