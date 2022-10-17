#include "update_t.h"

double update_t::evaluate_expression(simulator_state* state) const
{
    state->value_stack.clear();
    state->expression_stack.clear();
    
    update_expression* current = this->expression_;
    while (true)
    {
        while(current != nullptr)
        {
            
            state->expression_stack.push(current);
            state->expression_stack.push(current);

            // if(!current->is_leaf()) //only push twice if it has children
            //      this->expression_stack_->push(current);
            current = current->get_left();
        }
        if(state->expression_stack.is_empty())
        {
            break;
        }
        current = state->expression_stack.pop();
        
        if(!state->expression_stack.is_empty() && state->expression_stack.peak() == current)
        {
            current = current->get_right(&state->value_stack);
        }
        else
        {
            current->evaluate(state);
            current = nullptr;
        }
    }

    if(state->value_stack.count() == 0)
    {
        printf("Expression evaluation ended in no values! PANIC!\n");
        return 0;
    }
    return state->value_stack.pop();
}

update_t::update_t(const update_t* source, update_expression* expression)
{
    this->id_ = source->id_;
    this->variable_id_ = source->variable_id_;
    this->is_clock_update_ = source->is_clock_update_;
    this->expression_ = expression;
}

update_t::update_t(const int id, const int variable_id, const bool is_clock_update, update_expression* expression)
{
    this->id_ = id;
    this->variable_id_ = variable_id;
    this->is_clock_update_ = is_clock_update;
    this->expression_ = expression;
}

CPU GPU void update_t::apply_update(simulator_state* state) const
{
    const double value = evaluate_expression(state);
    if(this->is_clock_update_)
    {
        state->timers.at(this->variable_id_)->set_time(value);
    }
    else
    {
        //value is rounded correctly by adding 0.5. casting always rounds down.
        state->variables.at(this->variable_id_)->set_value(static_cast<int>(value + 0.5));  // NOLINT(bugprone-incorrect-roundings)
    }
}

CPU GPU void update_t::apply_temp_update(simulator_state* state) const
{
    const double value = evaluate_expression(state);
    if(this->is_clock_update_)
    {
        state->timers.at(this->variable_id_)->set_temp_time(value);
    }
    else
    {
        //value is rounded correctly by adding 0.5. casting always rounds down.
        state->variables.at(this->variable_id_)
            ->set_temp_value(static_cast<int>(value + 0.5));  // NOLINT(bugprone-incorrect-roundings)
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
        //TODO FIX THIS LATER
        state->variables.at(this->variable_id_)->reset_temp();  // NOLINT(bugprone-incorrect-roundings)
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
    update_expression* expression = nullptr;
    cudaMalloc(&expression, sizeof(update_expression));
    helper->free_list->push_back(expression);
    this->expression_->cuda_allocate(expression, helper);
    
    //  // = this->expression_->cuda_allocate(helper);
    // cuda_stack<double>* value_stack = this->value_stack_->cuda_allocate(helper);
    // cuda_stack<update_expression*>* expression_stack = this->expression_stack_->cuda_allocate(helper);

    const update_t copy = update_t(this, expression);
    cudaMemcpy(cuda, &copy, sizeof(update_t), cudaMemcpyHostToDevice);
}
