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
        state->variables.at(this->variable_id_)->set_time(value);  // NOLINT(bugprone-incorrect-roundings)
        printf("--------Variable id: %d | value: %f\n", this->variable_id_ , state->variables.at(this->variable_id_)->get_time());
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
            ->set_temp_time(value);  // NOLINT(bugprone-incorrect-roundings)
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

void update_t::pretty_print() const
{
    printf("Update id: %3d | Timer id: %3d\n", this->id_, this->variable_id_);
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


unsigned update_t::get_expression_depth() const
{
    return this->expression_->get_depth();
}
