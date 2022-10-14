#include "update_t.h"

int update_t::evaluate_expression(const lend_array<clock_timer_t>* timers, const lend_array<system_variable>* variables) const
{
    this->value_stack_->clear();
    if(this->value_stack_->max_size() <= 1)
    {
        this->expression_->evaluate(this->value_stack_, timers, variables);
        return this->value_stack_->pop();
    }
    this->expression_stack_->clear();


    update_expression* current = nullptr;
    while (true)
    {
        while(current != nullptr)
        {
            this->expression_stack_->push(current);
            this->expression_stack_->push(current);
            current = current->get_left();
        }
        if(this->expression_stack_->is_empty())
        {
            break;
        }
        current = this->expression_stack_->pop();
        if(!this->expression_stack_->is_empty() && this->expression_stack_->peak() == current)
        {
            current = current->get_right();
        }
        else
        {
            current->evaluate(this->value_stack_, timers, variables);
            current = nullptr;
        }
    }

    return this->value_stack_->pop();
}

update_t::update_t(const update_t* source, update_expression* expression,
        cuda_stack<int>* value_stack, cuda_stack<update_expression*>* evaluation_stack)
{
    this->id_ = source->id_;
    this->variable_id_ = source->variable_id_;
    this->is_clock_update_ = source->is_clock_update_;
    this->expression_ = expression;
    this->value_stack_ = value_stack;
    this->expression_stack_ = evaluation_stack;
}

update_t::update_t(const int id, const int timer_id, const bool is_clock_update, update_expression* expression)
{
    this->id_ = id;
    this->variable_id_ = timer_id;
    this->is_clock_update_ = is_clock_update;
    this->expression_ = expression;

    const unsigned int expression_depth = expression->get_depth();
    this->expression_stack_ = new cuda_stack<update_expression*>(expression_depth*2+1);
    this->value_stack_ = new cuda_stack<int>(expression_depth);
}

CPU GPU void update_t::apply_update(const lend_array<clock_timer_t>* timers, const lend_array<system_variable>* variables) const
{
    const double value = evaluate_expression(timers, variables);
    if(this->is_clock_update_)
    {
        timers->at(this->variable_id_)->set_time(value);
    }
    else
    {
        //value is rounded correctly by adding 0.5. casting always rounds down.
        variables->at(this->variable_id_)->set_value(static_cast<int>(value + 0.5));  // NOLINT(bugprone-incorrect-roundings)
    }
}

void update_t::accept(visitor* v)
{
    return;
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
    
     // = this->expression_->cuda_allocate(helper);
    cuda_stack<int>* value_stack = this->value_stack_->cuda_allocate(helper);
    cuda_stack<update_expression*>* expression_stack = this->expression_stack_->cuda_allocate(helper);

    const update_t copy = update_t(this, expression, value_stack, expression_stack);
    cudaMemcpy(cuda, &copy, sizeof(update_t), cudaMemcpyHostToDevice);
}
